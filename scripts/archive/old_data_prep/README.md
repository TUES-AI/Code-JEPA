# Archived data-prep scripts

These scripts produced the first proof-of-learning Code-JEPA data roots. They are kept for reproducibility only.

The current tokenizer-agnostic prep entrypoint is:

```bash
PYTHONPATH=src python scripts/prepare_data.py --help
```

Archived scripts:

- `prepare_codeparrot_python.py`: old CodeParrot whole-file/function extractor.
- `prepare_codesearchnet_python.py`: old CodeSearchNet function extractor.
- `augment_harder_triples.py`: old harder-negative-only augmentation pass.
