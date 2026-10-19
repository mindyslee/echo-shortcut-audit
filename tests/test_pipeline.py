import numpy as np
import pandas as pd
import pytest

from echoaudit.data import join, load_embeddings_txt, make_synthetic_filelist
from echoaudit.probes import is_categorical, probe
from echoaudit.removal import cost_curve, nuisance_design, regress_out
from echoaudit.report import AuditConfig, run_audit, to_markdown, write_report
from echoaudit.stratify import metadata_only_baseline, partial_r2, stratified_error


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(1)
    n, d = 600, 32
    ids = np.array([f"V{i:05d}" for i in range(n)])
    X = rng.normal(size=(n, d)).astype(np.float32)
    meta = make_synthetic_filelist(ids, X, seed=1, leak_strength=0.8)
    return join(ids, X, meta, source="synthetic")


def test_load_txt(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("A 1 2 3\nB 4 5 6\n")
    ids, X = load_embeddings_txt(p)
    assert list(ids) == ["A", "B"] and X.shape == (2, 3) and X.dtype == np.float32


def test_load_txt_rejects_ragged(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("A 1 2 3\nB 4 5\n")
    with pytest.raises(ValueError):
        load_embeddings_txt(p)


def test_join_drops_unmatched_and_bins_ef():
    ids = np.array(["a", "b", "c"])
    X = np.eye(3, dtype=np.float32)
    meta = pd.DataFrame({"FileName": ["a.avi", "c"], "EF": [35.0, 62.0]})
    ds = join(ids, X, meta)
    assert ds.n == 2 and list(ds.ids) == ["a", "c"]
    assert list(ds.meta["EF_bin"].astype(str)) == ["EF<40", "EF>=50"]
    assert ds.notes and "1 embeddings" in ds.notes[0]


def test_is_categorical():
    assert is_categorical(np.array([50, 60, 50, 30]))
    assert not is_categorical(np.arange(100).astype(float))
    assert is_categorical(np.array(["x", "y"]))


def test_probe_finds_planted_signal_and_null_is_flat(synth):
    r = probe(synth.X, synth.target("EF"), "EF", n_permutations=10, seed=0)
    assert r.kind == "regression"
    assert r.score > 0.3
    assert abs(r.null_mean) < 0.1
    assert r.p_value < 0.1

    c = probe(synth.X, synth.target("FPS"), "FPS", n_permutations=10, seed=0)
    assert c.kind == "classification"
    assert c.score > c.chance + 0.1
    assert c.p_value < 0.1


def test_decodable_gate_uses_z_not_only_p(synth):
    # 3 permutations floor p at 0.25; a strong signal must still count as decodable via z
    r = probe(synth.X, synth.target("EF"), "EF", n_permutations=3, seed=0)
    assert r.p_value >= 0.25 and r.decodable
    rng = np.random.default_rng(9)
    Xr = rng.normal(size=synth.X.shape).astype(np.float32)
    assert not probe(Xr, synth.target("EF"), "EF", n_permutations=3, seed=0).decodable


def test_probe_random_embedding_is_at_chance(synth):
    rng = np.random.default_rng(5)
    Xr = rng.normal(size=synth.X.shape).astype(np.float32)
    r = probe(Xr, synth.target("EF"), "EF", n_permutations=0)
    assert r.score < 0.05


def test_regress_out_kills_linear_signal(synth):
    N = nuisance_design(synth.meta, ["NumberOfFrames"])
    Xr = regress_out(synth.X, N)
    y = synth.target("NumberOfFrames").astype(float)
    # after residualising, every column is orthogonal to the nuisance
    corr = np.corrcoef(np.c_[Xr, y].T)[-1, :-1]
    assert np.abs(corr).max() < 1e-4


def test_cost_curve_removes_nuisance(synth):
    cc = cost_curve(
        synth.X,
        synth.target("FPS"),
        "FPS",
        {"EF": synth.target("EF")},
        n_iter=6,
        n_splits=3,
    )
    assert cc.nuisance_score[0] > cc.nuisance_chance + 0.1
    assert cc.nuisance_score[-1] < cc.nuisance_chance + 0.08
    assert len(cc.clinical["EF"]) == len(cc.iterations)
    assert cc.rank_removed[-1] > 0
    assert set(cc.random_control) == {"EF"}
    assert abs(cc.price_vs_random("EF")) < 0.15


def test_stratify_pieces(synth):
    nz = ["FPS", "NumberOfFrames"]
    b = metadata_only_baseline(synth.meta, "EF", nz, n_splits=3)
    assert set(b) >= {"linear_r2", "boosted_r2"}
    p = partial_r2(synth.X, synth.meta, "EF", nz, n_splits=3)
    assert p["r2_both"] >= p["r2_metadata"] - 0.05
    s = stratified_error(synth.X, synth.meta, "EF", nz, n_splits=3)
    assert {row["stratum"] for row in s} == set(nz)
    assert all(0 <= row["kruskal_p"] <= 1 for row in s)


def test_full_audit_and_report(synth, tmp_path):
    cfg = AuditConfig(n_splits=3, n_permutations=5, inlp_iterations=4)
    res = run_audit(synth, cfg, log=lambda s: None)
    # FrameHeight/FrameWidth are constant in the synthetic file and must be skipped
    assert "FrameHeight" in res["skipped_targets"]
    assert "EF" in res["probes"] and "FPS" in res["probes"]
    md = to_markdown(res)
    assert "Decodability" in md and "FPS" in md
    paths = write_report(res, tmp_path)
    assert paths["json"].exists() and paths["markdown"].exists() and paths["decodability"].exists()
