"""
Fine-tune ChemBERTa + RDKit descriptor fusion for molecular classification.
Configuration: JSON file (--config) plus CLI flags (CLI overrides JSON).
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

from utils_adapt import (
    MoleculeDataset,
    compute_descriptor_matrix,
    ChemBERTaDescriptorFusion,
    prepare_descriptors_with_gaussianized_pca,
)


def load_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def deep_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d and d[key] is not None else default


def train_epoch(model, dataloader, optimizer, criterion, device,
                gradient_accumulation_steps=1, use_amp=False, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_attentions = []
    optimizer.zero_grad()
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        desc = batch['desc'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast(amp_device, enabled=use_amp and device.type == 'cuda'):
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                desc=desc,
                labels=labels,
                output_attentions=True,
                return_dict=False
            )

            if len(result) == 2:
                logits, attentions = result
                loss = criterion(logits, labels)
            elif len(result) == 3:
                loss, logits, attentions = result
            else:
                loss, logits, attentions = result[0], result[1], result[2]

            cpu_attentions = {}
            if isinstance(attentions, dict):
                for k, v in attentions.items():
                    if isinstance(v, torch.Tensor):
                        cpu_attentions[k] = v.detach().cpu()
                    elif isinstance(v, list):
                        cpu_attentions[k] = [t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in v]
            elif isinstance(attentions, tuple):
                for i, attn in enumerate(attentions):
                    cpu_attentions[f'layer_{i}'] = attn.detach().cpu() if isinstance(attn, torch.Tensor) else attn

            del attentions, cpu_attentions
            loss = loss / gradient_accumulation_steps

        if use_amp and scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        del input_ids, attention_mask, desc, labels, logits, loss
        if device.type == 'cuda' and (step + 1) % 20 == 0:
            torch.cuda.empty_cache()

    if len(dataloader) % gradient_accumulation_steps != 0:
        if use_amp and scaler is not None and device.type == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, all_attentions


def eval_epoch(model, dataloader, criterion, device, use_amp=False, output_sequence_output=False,
               save_seq_output_path=None, grad_clip=1.0, seq_chunk_batches=10):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_attentions = []
    seq_output_chunks = []
    current_chunk = []
    chunk_size = seq_chunk_batches
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            desc = batch['desc'].to(device)
            labels = batch['label'].to(device)
            loss = None

            with torch.amp.autocast(amp_device, enabled=use_amp and device.type == 'cuda'):
                if output_sequence_output:
                    result = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        desc=desc,
                        output_attentions=False,
                        output_sequence_output=True,
                        return_dict=False
                    )
                    if len(result) == 3:
                        logits, attentions, sequence_output = result
                    elif len(result) == 4:
                        loss, logits, attentions, sequence_output = result
                    else:
                        logits = result[0]
                        attentions = result[1] if len(result) > 1 else None
                        sequence_output = result[2] if len(result) > 2 else None

                    if sequence_output is not None:
                        seq_out_np = sequence_output.detach().cpu().numpy()
                        del sequence_output
                        if save_seq_output_path:
                            current_chunk.append(seq_out_np)
                            if len(current_chunk) >= chunk_size:
                                merged_chunk = np.concatenate(current_chunk, axis=0)
                                seq_output_chunks.append(merged_chunk)
                                current_chunk = []
                                del merged_chunk
                else:
                    result = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        desc=desc,
                        output_attentions=False,
                        return_dict=False
                    )
                    if len(result) == 2:
                        logits, attentions = result
                    elif len(result) >= 3:
                        if len(result) == 3:
                            loss, logits, attentions = result
                        else:
                            logits, attentions = result[0], result[1]
                    else:
                        logits = result[0]
                        attentions = result[1] if len(result) > 1 else None

                cpu_attentions = {}
                if attentions is not None:
                    if isinstance(attentions, dict):
                        for k, v in attentions.items():
                            if isinstance(v, torch.Tensor):
                                cpu_attentions[k] = v.cpu()
                            elif isinstance(v, list):
                                cpu_attentions[k] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in v]
                    elif isinstance(attentions, tuple):
                        for i, attn in enumerate(attentions):
                            cpu_attentions[f'layer_{i}'] = attn.cpu() if isinstance(attn, torch.Tensor) else attn

                all_attentions.append(cpu_attentions)
                del attentions, cpu_attentions

                if loss is None:
                    loss = criterion(logits, labels)

            total_loss += loss.item()
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            del input_ids, attention_mask, desc, labels, logits, loss
            if device.type == 'cuda' and (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

    if output_sequence_output and save_seq_output_path:
        if current_chunk:
            merged_chunk = np.concatenate(current_chunk, axis=0)
            seq_output_chunks.append(merged_chunk)
        if seq_output_chunks:
            print(f"Saving sequence_output to {save_seq_output_path}...")
            final_seq_output = np.concatenate(seq_output_chunks, axis=0)
            np.save(save_seq_output_path, final_seq_output)
            print(f"Sequence output saved (shape: {final_seq_output.shape})")
            del final_seq_output, seq_output_chunks

    avg_loss = total_loss / max(len(dataloader), 1)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return avg_loss, all_predictions, all_labels, all_attentions


def test_model(model, test_loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            desc = batch['desc'].to(device)
            labels = batch['label'].to(device)

            with torch.amp.autocast(amp_device, enabled=use_amp and device.type == 'cuda'):
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    desc=desc,
                    output_attentions=False,
                    return_dict=False
                )
                if len(result) == 1:
                    logits = result[0]
                elif len(result) == 2:
                    logits, _ = result
                else:
                    logits = result[0]

                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            del input_ids, attention_mask, desc, labels, logits, loss, probs, preds

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    accuracy = accuracy_score(all_labels, all_predictions)

    if all_probs.shape[1] == 2:
        y_probs = all_probs[:, 1]
    else:
        y_probs = np.max(all_probs, axis=1)

    try:
        auroc = roc_auc_score(all_labels, y_probs)
    except ValueError as e:
        print(f"Warning: could not compute AUROC: {e}")
        auroc = None

    avg_loss = total_loss / max(len(test_loader), 1)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'probs': all_probs,
        'accuracy': accuracy,
        'auroc': auroc
    }


def generate_scaffold(smiles, include_chirality=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    except Exception:
        return None


def scaffold_split(df, smiles_col='SMILES', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1"

    print("Scaffold splitting...")
    print(f"Target ratios — train: {train_ratio:.1%}, val: {val_ratio:.1%}, test: {test_ratio:.1%}")

    scaffolds = {}
    scaffold_to_indices = defaultdict(list)

    for idx, smiles in enumerate(tqdm(df[smiles_col], desc="Scaffolds")):
        scaffold = generate_scaffold(smiles)
        if scaffold is None:
            scaffold = smiles
        scaffolds[idx] = scaffold
        scaffold_to_indices[scaffold].append(idx)

    scaffold_groups = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"Unique scaffolds: {len(scaffold_groups)}")
    print(
        f"Scaffold group sizes — max: {len(scaffold_groups[0][1])}, "
        f"min: {len(scaffold_groups[-1][1])}, "
        f"mean: {np.mean([len(g[1]) for g in scaffold_groups]):.2f}"
    )

    train_indices, val_indices, test_indices = [], [], []
    train_count = val_count = test_count = 0
    total_count = len(df)
    target_train = int(total_count * train_ratio)
    target_val = int(total_count * val_ratio)
    target_test = total_count - target_train - target_val
    np.random.seed(random_state)

    for scaffold, indices in tqdm(scaffold_groups, desc="Assign scaffolds"):
        current_train_ratio = train_count / total_count if total_count > 0 else 0
        current_val_ratio = val_count / total_count if total_count > 0 else 0

        if train_count < target_train and (current_train_ratio < train_ratio or val_count >= target_val):
            train_indices.extend(indices)
            train_count += len(indices)
        elif val_count < target_val and current_val_ratio < val_ratio:
            val_indices.extend(indices)
            val_count += len(indices)
        else:
            test_indices.extend(indices)
            test_count += len(indices)

    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    test_indices = np.array(sorted(test_indices))

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Scaffold split summary:")
    print("=" * 60)
    print(f"Train: {len(train_df)} ({len(train_df) / len(df):.2%})")
    print(f"Val:   {len(val_df)} ({len(val_df) / len(df):.2%})")
    print(f"Test:  {len(test_df)} ({len(test_df) / len(df):.2%})")
    print(f"Train scaffolds: {len(set(scaffolds[i] for i in train_indices))}")
    print(f"Val scaffolds:   {len(set(scaffolds[i] for i in val_indices))}")
    print(f"Test scaffolds:  {len(set(scaffolds[i] for i in test_indices))}")
    print("=" * 60 + "\n")

    return train_df, val_df, test_df, train_indices, val_indices, test_indices


def parse_args(argv: Optional[list] = None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config (default: config.json next to this script, if present)',
    )
    pre_args, remaining = pre.parse_known_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'config.json')
    config_path = pre_args.config
    if config_path is None and os.path.isfile(default_config):
        config_path = default_config
    cfg = load_config(config_path) if config_path else {}

    def g(key, fallback):
        return deep_get(cfg, key, fallback)

    p = argparse.ArgumentParser(
        description='Fine-tune ChemBERTa + descriptor fusion for classification.',
        parents=[pre],
    )
    p.add_argument('--data-csv', type=str, default=g('data_csv', ''), help='Single CSV with SMILES + label (if not load-from-files)')
    p.add_argument('--smiles-col', type=str, default=g('smiles_col', 'SMILES'))
    p.add_argument('--label-col', type=str, default=g('label_col', 'label'))
    p.add_argument('--pretrained-model-name', type=str, default=g('pretrained_model_name', 'DeepChem/ChemBERTa-100M-MLM'))
    p.add_argument('--batch-size', type=int, default=g('batch_size', 128))
    p.add_argument('--epochs', type=int, default=g('epochs', 200))
    p.add_argument('--lr', type=float, default=g('lr', 2e-5))
    p.add_argument('--max-length', type=int, default=g('max_length', 512))
    p.add_argument('--use-cross-attention', action=argparse.BooleanOptionalAction, default=g('use_cross_attention', True))
    p.add_argument('--freeze-text', action=argparse.BooleanOptionalAction, default=g('freeze_text', True))
    p.add_argument('--fusion-hidden', type=int, default=g('fusion_hidden', 512))
    p.add_argument('--gradient-accumulation-steps', type=int, default=g('gradient_accumulation_steps', 8))
    p.add_argument('--use-gaussianized-pca', action=argparse.BooleanOptionalAction, default=g('use_gaussianized_pca', True))
    p.add_argument(
        '--pca-n-components',
        type=int,
        default=g('pca_n_components', None),
        help='PCA n_components (omit or use config null for full rank)',
    )
    p.add_argument('--pca-whiten', action=argparse.BooleanOptionalAction, default=g('pca_whiten', True))
    p.add_argument('--pca-n-quantiles', type=int, default=g('pca_n_quantiles', 1000))
    p.add_argument('--random-state', type=int, default=g('random_state', 42))
    p.add_argument('--device', type=str, default=g('device', 'cuda'), choices=['cuda', 'cpu'])
    p.add_argument('--cuda-device-index', type=int, default=g('cuda_device_index', 0))
    p.add_argument('--output-dir', type=str, default=g('output_dir', './output'))
    p.add_argument('--load-from-files', action=argparse.BooleanOptionalAction, default=g('load_from_files', True))
    p.add_argument('--data-dir', type=str, default=g('data_dir', ''))
    p.add_argument('--train-csv', type=str, default=g('train_csv', 'train.csv'))
    p.add_argument('--val-csv', type=str, default=g('val_csv', 'valid.csv'))
    p.add_argument('--test-csv', type=str, default=g('test_csv', 'test.csv'))
    p.add_argument('--use-scaffold-split', action=argparse.BooleanOptionalAction, default=g('use_scaffold_split', True))
    p.add_argument('--train-ratio', type=float, default=g('train_ratio', 0.8))
    p.add_argument('--val-ratio', type=float, default=g('val_ratio', 0.1))
    p.add_argument('--test-ratio', type=float, default=g('test_ratio', 0.1))
    p.add_argument('--output-sequence-output', action=argparse.BooleanOptionalAction, default=g('output_sequence_output', True))
    p.add_argument('--save-seq-output-every-epoch', action=argparse.BooleanOptionalAction, default=g('save_seq_output_every_epoch', False))
    p.add_argument('--use-early-stopping', action=argparse.BooleanOptionalAction, default=g('use_early_stopping', True))
    p.add_argument('--early-stopping-patience', type=int, default=g('early_stopping_patience', 30))
    p.add_argument('--num-workers', type=int, default=g('num_workers', 2))
    p.add_argument('--num-classes', type=int, default=g('num_classes', 2))
    p.add_argument('--warmup-steps', type=int, default=g('warmup_steps', 100))
    p.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=g('use_amp', True))
    p.add_argument('--grad-clip', type=float, default=g('grad_clip', 1.0))
    p.add_argument('--set-cuda-alloc-expandable', action=argparse.BooleanOptionalAction, default=g('set_cuda_alloc_expandable', True))
    p.add_argument('--seq-chunk-batches', type=int, default=g('seq_chunk_batches', 10), help='Batches per chunk when saving val sequence outputs')

    args = p.parse_args(remaining)

    if args.load_from_files and not args.data_dir:
        raise ValueError('--data-dir is required when --load-from-files is true')
    if not args.load_from_files and not args.data_csv:
        raise ValueError('--data-csv is required when --no-load-from-files')

    return args


def resolve_device(args) -> torch.device:
    if args.device == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(f'cuda:{args.cuda_device_index}')


def main():
    args = parse_args()

    if args.set_cuda_alloc_expandable and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    device = resolve_device(args)
    print(f"Device: {device}")

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    attentions_dir = os.path.join(args.output_dir, 'attentions')
    os.makedirs(attentions_dir, exist_ok=True)

    if args.load_from_files:
        print("Loading train/val/test CSVs from data directory...")
        train_csv_path = os.path.join(args.data_dir, args.train_csv)
        val_csv_path = os.path.join(args.data_dir, args.val_csv)
        test_csv_path = os.path.join(args.data_dir, args.test_csv)
        for path, name in [(train_csv_path, args.train_csv), (val_csv_path, args.val_csv), (test_csv_path, args.test_csv)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    else:
        df = pd.read_csv(args.data_csv)
        if args.use_scaffold_split:
            print("Scaffold split from single CSV...")
            train_df, val_df, test_df, _, _, _ = scaffold_split(
                df,
                smiles_col=args.smiles_col,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.random_state,
            )
        else:
            print("Random split from single CSV...")
            train_val_df, test_df, _, _ = train_test_split(
                df, np.arange(len(df)), test_size=args.test_ratio, random_state=args.random_state
            )
            train_df, val_df, _, _ = train_test_split(
                train_val_df,
                np.arange(len(train_val_df)),
                test_size=args.val_ratio / (1.0 - args.test_ratio),
                random_state=args.random_state,
            )

    if args.use_gaussianized_pca:
        print("Gaussianized PCA for descriptors...")
        train_desc, val_desc, test_desc, transformer = prepare_descriptors_with_gaussianized_pca(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            smiles_col=args.smiles_col,
            n_components=args.pca_n_components,
            whiten=args.pca_whiten,
            n_quantiles=args.pca_n_quantiles,
            random_state=args.random_state,
            verbose=True,
        )
        joblib.dump(transformer, os.path.join(args.output_dir, 'gaussianized_pca_transformer.pkl'))
        print(f"Saved transformer to {os.path.join(args.output_dir, 'gaussianized_pca_transformer.pkl')}")
        print(f"Descriptor dim after transform: {train_desc.shape[1]}")
    else:
        print("Computing descriptors + StandardScaler (per train/val/test split)...")
        train_desc_df, _ = compute_descriptor_matrix(train_df[args.smiles_col].tolist(), verbose=True)
        train_desc_df = train_desc_df.fillna(train_desc_df.mean())
        train_desc_array = train_desc_df.values.astype(np.float32)

        val_desc_df, _ = compute_descriptor_matrix(val_df[args.smiles_col].tolist(), verbose=True)
        val_desc_df = val_desc_df.fillna(train_desc_df.mean())
        val_desc_array = val_desc_df.values.astype(np.float32)

        test_desc_df, _ = compute_descriptor_matrix(test_df[args.smiles_col].tolist(), verbose=True)
        test_desc_df = test_desc_df.fillna(train_desc_df.mean())
        test_desc_array = test_desc_df.values.astype(np.float32)

        scaler = StandardScaler()
        train_desc = scaler.fit_transform(train_desc_array)
        val_desc = scaler.transform(val_desc_array)
        test_desc = scaler.transform(test_desc_array)
        joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))
        transformer = None

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, use_fast=False)

    train_dataset = MoleculeDataset(
        train_df, smiles_col=args.smiles_col, label_col=args.label_col,
        tokenizer=tokenizer, max_length=args.max_length, desc_array=train_desc, transformer=None,
    )
    val_dataset = MoleculeDataset(
        val_df, smiles_col=args.smiles_col, label_col=args.label_col,
        tokenizer=tokenizer, max_length=args.max_length, desc_array=val_desc, transformer=None,
    )
    test_dataset = MoleculeDataset(
        test_df, smiles_col=args.smiles_col, label_col=args.label_col,
        tokenizer=tokenizer, max_length=args.max_length, desc_array=test_desc, transformer=None,
    )
    print(f"Datasets — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    descriptor_dim = train_desc.shape[1]
    print(f"Model descriptor dim: {descriptor_dim}")
    model = ChemBERTaDescriptorFusion(
        pretrained_model_name=args.pretrained_model_name,
        descriptor_dim=descriptor_dim,
        fusion_hidden=args.fusion_hidden,
        freeze_text=args.freeze_text,
        use_cross_attention=args.use_cross_attention,
        num_classes=args.num_classes,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (FP16) enabled")
    else:
        print("Mixed precision disabled (CPU or --no-use-amp)")

    effective_steps = max(len(train_loader) // max(args.gradient_accumulation_steps, 1), 1)
    total_steps = effective_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(device.index if device.index is not None else 0)
        print(f"GPU memory: {props.total_memory / 1024 ** 3:.2f} GB")

    print("Training...")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        scheduler.step()

        is_last_epoch = (epoch + 1 == args.epochs)
        should_save_seq = args.output_sequence_output and (args.save_seq_output_every_epoch or is_last_epoch)

        if should_save_seq:
            val_seq_output_path = os.path.join(attentions_dir, f"val_sequence_output_epoch_{epoch + 1}.npy")
            val_loss, val_predictions, val_labels, val_attentions = eval_epoch(
                model, val_loader, criterion, device,
                use_amp=use_amp,
                output_sequence_output=True,
                save_seq_output_path=val_seq_output_path,
                grad_clip=args.grad_clip,
                seq_chunk_batches=args.seq_chunk_batches,
            )
        else:
            val_loss, val_predictions, val_labels, val_attentions = eval_epoch(
                model, val_loader, criterion, device,
                use_amp=use_amp,
                output_sequence_output=False,
                grad_clip=args.grad_clip,
            )

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if args.use_early_stopping:
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_epoch = epoch + 1
                early_stopping_counter = 0
                print(f"Val loss improved by {improvement:.6f}; saving best model (epoch {best_epoch})...")
                model.save_pretrained(args.output_dir)
            else:
                early_stopping_counter += 1
                print(f"No val improvement ({early_stopping_counter}/{args.early_stopping_patience})")
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"\nEarly stopping after {args.early_stopping_patience} epochs without improvement.")
                    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
                    break
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model (val loss {val_loss:.4f})...")
                model.save_pretrained(args.output_dir)

    print("\nTraining finished.")
    if args.use_early_stopping:
        print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch}), ran {epoch + 1}/{args.epochs} epochs")
    else:
        print(f"Best val loss: {best_val_loss:.4f}")

    if args.output_sequence_output and not args.save_seq_output_every_epoch:
        print("\nFinal eval: saving validation sequence_output...")
        final_seq_output_path = os.path.join(attentions_dir, "val_sequence_output_final.npy")
        eval_epoch(
            model, val_loader, criterion, device,
            use_amp=use_amp,
            output_sequence_output=True,
            save_seq_output_path=final_seq_output_path,
            grad_clip=args.grad_clip,
            seq_chunk_batches=args.seq_chunk_batches,
        )
        print(f"Saved under: {attentions_dir}")

    print("\n" + "=" * 50)
    print("Test set evaluation")
    print("=" * 50)
    try:
        best_model = ChemBERTaDescriptorFusion.from_pretrained(args.output_dir)
        best_model.to(device)
        best_model.eval()
        model = best_model
        print("Loaded best saved weights")
    except Exception:
        print("Using in-memory weights for evaluation")
        model.eval()

    test_results = test_model(model, test_loader, criterion, device, use_amp=use_amp)

    print("\n" + "=" * 50)
    print("Test metrics")
    print("=" * 50)
    print(f"Test loss: {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy'] * 100:.2f}%)")
    if test_results['auroc'] is not None:
        print(f"AUROC: {test_results['auroc']:.4f}")
    else:
        print("AUROC: N/A")
    print("=" * 50)

    test_results_path = os.path.join(args.output_dir, 'test_results.txt')
    with open(test_results_path, 'w', encoding='utf-8') as f:
        f.write("Test set results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test loss: {test_results['loss']:.4f}\n")
        f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        if test_results['auroc'] is not None:
            f.write(f"AUROC: {test_results['auroc']:.4f}\n")
        else:
            f.write("AUROC: N/A\n")

    print(f"\nWrote: {test_results_path}")

    num_classes = test_results['probs'].shape[1]
    pred_dict = {
        'true_label': test_results['labels'],
        'predicted_label': test_results['predictions'],
    }
    for i in range(num_classes):
        pred_dict[f'prob_class_{i}'] = test_results['probs'][:, i]

    test_predictions_path = os.path.join(args.output_dir, 'test_predictions.csv')
    pd.DataFrame(pred_dict).to_csv(test_predictions_path, index=False)
    print(f"Wrote: {test_predictions_path}")


if __name__ == '__main__':
    main()
