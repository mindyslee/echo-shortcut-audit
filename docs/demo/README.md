# Reference demo output (SYNTHETIC labels)

Produced by

    echoaudit demo --embeddings echonet_embeddings.txt --out docs/demo --permutations 10 --leak 0.8

on the real 10,030 × 400 R3D embeddings paired with a **synthetic** `FileList` whose EF and
acquisition fields are random projections of the embedding with a planted leak of strength 0.8.
It exists to show the audit recovers a known leak and prices it sensibly; none of the numbers
say anything about the actual EchoNet-Dynamic labels. See `audit.md` for the tables and the two
figures for the decodability bars and INLP cost curves.
