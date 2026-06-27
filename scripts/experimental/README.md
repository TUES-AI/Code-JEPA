# Experimental trainers

These scripts are simple research prototypes kept for reproducibility and comparison.

They are not the optimized training path:
- they tokenize raw code strings online;
- they join Parquet `views` and `triples` in Python during training;
- they are useful for quick experiments, not full-data throughput.

Use `scripts/tokenize_jepa_triples.py` plus `scripts/train_siamese_bpe_jepa.py` for the production single-GPU path.
