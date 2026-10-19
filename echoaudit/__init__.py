"""echoaudit: shortcut / nuisance-attribute audit for frozen echocardiography embeddings.

Pipeline
--------
1. ``data``      load an embedding matrix and an EchoNet-style ``FileList.csv``; join on video id.
2. ``probes``    linear-probe the embedding for clinical targets (EF, ESV, EDV) and
                 acquisition nuisances (FPS, frame height/width, number of frames), with
                 cross-validation, a permutation null and a random-projection control.
3. ``removal``   erase nuisance directions (regress-out or iterative null-space projection)
                 and re-probe the clinical targets to price the shortcut.
4. ``stratify``  metadata-only EF baseline and EF error stratified by acquisition bins.
5. ``report``    collect everything into JSON + Markdown + figures.
"""

__version__ = "0.1.0"
