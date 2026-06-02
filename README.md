# SN-BERT multimodal classification (SMILES + RDKit descriptors)

Fine-tune a **SN-BERT** encoder fused with **RDKit molecular descriptors**, with optional **Gaussianized PCA** preprocessing for descriptor vectors. This repository is a cleaned, configurable packaging of the original training and model code: paths and hyperparameters are **not hard-coded**; use **`config.json`** and/or **CLI flags** (flags override JSON when provided).

## Features

- **SN-BERTDescriptorFusion**: cross-attention or concat fusion between SMILES embeddings and descriptor vectors.
- **GaussianizedPCATransformer**: quantile (normal) transform + PCA, fit on train only.
- **Training script**: mixed precision (CUDA), gradient accumulation, optional early stopping, scaffold or random split from a single CSV, or separate `train.csv` / `valid.csv` / `test.csv`.
- **Saving**: `safetensors` fusion weights, `text_encoder/` subfolder, `model_config.json`, optional validation `sequence_output` `.npy` dumps.

## Requirements

- Python **3.9+** (uses `argparse.BooleanOptionalAction` for `--flag` / `--no-flag`).
- PyTorch with CUDA (optional; CPU is supported with `--device cpu` and `--no-use-amp`).
- [RDKit](https://www.rdkit.org/) (via `rdkit-pypi` on pip).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

1. Copy and edit **`config.json`** (or pass `--config /path/to/config.json`). Set at minimum:

   - **`chemberta_model_name`**: Hugging Face model id or local path (e.g. `DeepChem/ChemBERTa-100M-MLM` or your self-pretrained folder).
   - Either **`load_from_files`: true** and **`data_dir`** pointing to a folder containing `train.csv`, `valid.csv`, `test.csv` (names configurable),  
   - or **`load_from_files`: false**, **`data_csv`** path, and split options (`--use-scaffold-split` / `--no-use-scaffold-split`).

2. Run training from the directory that contains `adapt_SNbert.py` and `utils_adapt.py`:

```bash
python adapt_SNbert.py --config config.json
```

CLI examples (override JSON):

```bash
python adapt_SNbert.py --config config.json --epochs 50 --lr 3e-5 --output-dir ./runs/exp1

python adapt_SNbert.py --no-load-from-files --data-csv ./data/all.csv
```

When `--no-load-from-files` is set, **`--data-csv`** is required; **`--data-dir`** is not used.

```bash
python adapt_SNbert.py --load-from-files --data-dir /path/to/splits --chemberta-model-name /path/to/chemberta
```

## `config.json` reference

| Key | Description |
|-----|-------------|
| `data_csv` | Single CSV path if `load_from_files` is false |
| `smiles_col`, `label_col` | Column names |
| `chemberta_model_name` | Transformers model name or path |
| `batch_size`, `epochs`, `lr`, `max_length` | Training hyperparameters |
| `use_cross_attention`, `freeze_text` | Fusion / encoder freezing |
| `fusion_hidden`, `num_classes` | Model head |
| `gradient_accumulation_steps` | Gradient accumulation |
| `use_gaussianized_pca`, `pca_n_components`, `pca_whiten`, `pca_n_quantiles` | Descriptor preprocessing |
| `device` | `cuda` or `cpu` |
| `cuda_device_index` | GPU index when `device` is `cuda` |
| `output_dir` | Checkpoints, metrics, scaler/PCA artifacts |
| `load_from_files`, `data_dir`, `train_csv`, `val_csv`, `test_csv` | Data layout |
| `use_scaffold_split`, `train_ratio`, `val_ratio`, `test_ratio` | Splitting (single CSV) |
| `output_sequence_output`, `save_seq_output_every_epoch`, `seq_chunk_batches` | Optional `.npy` sequence dumps |
| `use_early_stopping`, `early_stopping_patience` | Early stopping |
| `num_workers`, `warmup_steps`, `use_amp`, `grad_clip` | DataLoader / scheduler / AMP |
| `random_state` | RNG seed for splits and PCA |
| `set_cuda_alloc_expandable` | Sets `PYTORCH_CUDA_ALLOC_CONF` if unset |

Run `python adapt_SNbert.py --help` for the full CLI list.

## Outputs

Under **`output_dir`** (default `./output`):

- Best fusion checkpoint: `model.safetensors`, `model_config.json`, `text_encoder/`, tokenizer files.
- `gaussianized_pca_transformer.pkl` or `scaler.pkl` depending on preprocessing.
- `test_results.txt`, `test_predictions.csv`.
- Optional: `attentions/val_sequence_output_*.npy`.

## License

Use and redistribute according to your institution’s policies and the licenses of **SN-BERT**, **Hugging Face Transformers**, and **RDKit**.

## Citation

If you use this code in research, please cite the original SN-BERT / related work you fine-tuned from, and reference this repository as appropriate.
