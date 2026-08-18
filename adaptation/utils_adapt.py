# Multimodal ChemBERTa + RDKit descriptors (Gaussianized PCA utilities and fusion model)
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file, load_file

try:
    from transformers.modeling_outputs import ModelOutput
except ImportError:
    from transformers.file_utils import ModelOutput

from dataclasses import dataclass

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------
# 1) RDKit descriptor calculator
# ----------------------------
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NOCount",
]


def compute_descriptors(smiles: str) -> Optional[List[float]]:
    """Compute a fixed set of RDKit descriptors for a SMILES. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        vals = []
        vals.append(Descriptors.MolWt(mol))
        vals.append(Descriptors.MolLogP(mol))
        vals.append(rdMolDescriptors.CalcNumHBD(mol))
        vals.append(rdMolDescriptors.CalcNumHBA(mol))
        vals.append(rdMolDescriptors.CalcTPSA(mol))
        vals.append(Descriptors.NumRotatableBonds(mol))
        vals.append(Descriptors.NOCount(mol))
        return vals
    except Exception:
        return None


def compute_descriptor_matrix(smiles_list: List[str], names=DESCRIPTOR_NAMES, verbose=True):
    rows = []
    valid_idx = []
    it = enumerate(smiles_list)
    if verbose:
        it = tqdm(it, total=len(smiles_list), desc="Computing descriptors")
    for i, s in it:
        vals = compute_descriptors(s)
        if vals is None:
            rows.append([np.nan] * len(names))
        else:
            rows.append(vals)
            valid_idx.append(i)
    df = pd.DataFrame(rows, columns=names)
    return df, valid_idx


# ----------------------------
# 1.5) Gaussianized PCA Transformer
# ----------------------------
class GaussianizedPCATransformer(BaseEstimator, TransformerMixin):
    """
    Quantile-transform each feature toward a normal distribution, then PCA (optional whiten).
    """
    def __init__(self, n_components=None, whiten=True, output_distribution='normal',
                 n_quantiles=1000, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self.quantile_transformer = QuantileTransformer(
            output_distribution=output_distribution,
            n_quantiles=n_quantiles,
            random_state=random_state
        )
        self.pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            self._col_medians = np.nanmedian(X, axis=0)
            X = X.copy()
            for col_idx in range(X.shape[1]):
                X[nan_mask[:, col_idx], col_idx] = self._col_medians[col_idx]
        else:
            self._col_medians = np.median(X, axis=0)

        self.quantile_transformer.fit(X)
        X_gaussianized = self.quantile_transformer.transform(X)
        self.pca.fit(X_gaussianized)
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            if not hasattr(self, '_col_medians'):
                self._col_medians = np.nanmedian(X, axis=0)
            X = X.copy()
            for col_idx in range(X.shape[1]):
                X[nan_mask[:, col_idx], col_idx] = self._col_medians[col_idx]

        X_gaussianized = self.quantile_transformer.transform(X)
        return self.pca.transform(X_gaussianized)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if hasattr(self.pca, 'n_components_'):
            n_out = self.pca.n_components_
        elif hasattr(self.pca, 'components_'):
            n_out = self.pca.components_.shape[0]
        else:
            n_out = self.n_components if self.n_components is not None else 9
        return [f'PC{i+1}' for i in range(n_out)]

    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_ if hasattr(self.pca, 'explained_variance_ratio_') else None


# ----------------------------
# 2) Dataset
# ----------------------------
class MoleculeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, smiles_col="SMILES", label_col="label",
                 tokenizer=None, max_length=128, desc_array=None,
                 transformer: Optional[TransformerMixin] = None):
        self.df = df.reset_index(drop=True)
        self.smiles = self.df[smiles_col].tolist()
        self.labels = self.df[label_col].tolist() if label_col in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.desc_array = desc_array
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        toks = self.tokenizer(s, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        item = {
            'input_ids': toks['input_ids'].squeeze(0),
            'attention_mask': toks['attention_mask'].squeeze(0),
        }
        if self.desc_array is not None:
            desc = self.desc_array[idx].astype(np.float32)
            if self.transformer is not None:
                desc = self.transformer.transform(desc.reshape(1, -1)).reshape(-1).astype(np.float32)
            item['desc'] = torch.from_numpy(desc)
        if self.labels is not None:
            item['label'] = torch.tensor(
                self.labels[idx],
                dtype=torch.float if isinstance(self.labels[0], float) else torch.long
            )
        return item


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    sequence_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# ----------------------------
# 3) Fusion Model
# ----------------------------
class CrossAttentionFusion(nn.Module):
    """Cross-attention: text as query, descriptor as key/value."""
    def __init__(self, text_dim, desc_dim, hidden_dim=256, n_heads=6):
        super().__init__()
        self.desc_proj = nn.Linear(desc_dim, text_dim)
        self.mha = nn.MultiheadAttention(embed_dim=text_dim, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, text_vec, desc_vec):
        if text_vec.dim() == 2:
            q = text_vec.unsqueeze(1)
        else:
            q = text_vec
        k_v = self.desc_proj(desc_vec).unsqueeze(1)
        attn_out, attn_weights = self.mha(query=q, key=k_v, value=k_v)
        attn_out = attn_out.squeeze(1)
        out = self.mlp(attn_out)
        return out, attn_weights


class ChemBERTaDescriptorFusion(nn.Module):
    def __init__(
        self,
        pretrained_model_name='ChemBERTa-100M-MLM',
        descriptor_dim=9,
        fusion_hidden=256,
        freeze_text=False,
        use_cross_attention=False,
        text_proj_dim=256,
        num_classes=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        text_dim = self.text_encoder.config.hidden_size

        self.freeze_text = freeze_text
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                text_dim=text_dim, desc_dim=descriptor_dim, hidden_dim=fusion_hidden
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_hidden),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_hidden // 2, num_classes)
            )
        else:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, text_proj_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.desc_proj = nn.Sequential(
                nn.Linear(descriptor_dim, text_proj_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            combined_dim = text_proj_dim + text_proj_dim // 2
            self.fusion_mlp = nn.Sequential(
                nn.Linear(combined_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.ReLU()
            )
            self.classifier = nn.Linear(fusion_hidden // 2, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        desc,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        output_sequence_output=False,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else True

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if not isinstance(desc, torch.Tensor):
            if isinstance(desc, np.ndarray):
                desc = torch.from_numpy(desc).float()
            else:
                desc = torch.tensor(desc, dtype=torch.float32)

        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        desc = desc.to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if desc.dim() == 1:
            desc = desc.unsqueeze(0)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs.last_hidden_state

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            text_vec = outputs.pooler_output
        else:
            text_vec = sequence_output[:, 0, :]

        if self.use_cross_attention:
            fused, _ = self.cross_attn(text_vec, desc)
            logits = self.classifier(fused)
        else:
            t = self.text_proj(text_vec)
            d = self.desc_proj(desc)
            combined = torch.cat([t, d], dim=-1)
            h = self.fusion_mlp(combined)
            logits = self.classifier(h)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            if self.num_classes == 1:
                from torch.nn import MSELoss
                loss_fct = MSELoss()
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.long().view(-1))

        all_attentions = outputs.attentions if output_attentions else None

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (outputs.hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            if output_sequence_output:
                output = output + (sequence_output,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            sequence_output=sequence_output if output_sequence_output else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        text_encoder_dir = os.path.join(save_directory, "text_encoder")
        os.makedirs(text_encoder_dir, exist_ok=True)
        try:
            self.text_encoder.save_pretrained(text_encoder_dir)
            print(f"  - Text encoder saved to: {text_encoder_dir}")
        except Exception as e:
            print(f"  Warning: could not save text encoder weights: {e}")

        if self.use_cross_attention:
            descriptor_dim = self.cross_attn.desc_proj.in_features
            fusion_hidden = (
                self.classifier[0].normalized_shape[0]
                if isinstance(self.classifier, nn.Sequential) else 256
            )
            num_classes = self.classifier[-1].out_features if isinstance(self.classifier, nn.Sequential) else 2
        else:
            descriptor_dim = self.desc_proj[0].in_features if isinstance(self.desc_proj, nn.Sequential) else 9
            fusion_hidden = 256
            num_classes = self.classifier.out_features if isinstance(self.classifier, nn.Linear) else 1

        model_config = {
            'descriptor_dim': descriptor_dim,
            'fusion_hidden': fusion_hidden,
            'freeze_text': self.freeze_text,
            'use_cross_attention': self.use_cross_attention,
            'text_proj_dim': 256,
            'num_classes': num_classes,
            'pretrained_model_name': getattr(self.text_encoder.config, 'name_or_path', 'ChemBERTa-100M-MLM')
        }
        with open(os.path.join(save_directory, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)

        state_dict = self.state_dict()
        save_file(state_dict, os.path.join(save_directory, 'model.safetensors'))

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                getattr(self.text_encoder.config, 'name_or_path', 'ChemBERTa-100M-MLM')
            )
            tokenizer.save_pretrained(save_directory)
            print("  - Tokenizer config saved")
        except Exception as e:
            print(f"  Warning: could not save tokenizer: {e}")

        print(f"Model saved to: {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        config_path = os.path.join(save_directory, 'model_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing model config: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)

        pretrained_model_name = model_config.get('pretrained_model_name', 'ChemBERTa-100M-MLM')

        text_encoder_dir = os.path.join(save_directory, "text_encoder")
        try:
            if os.path.isdir(text_encoder_dir):
                print(f"Loading text encoder from {text_encoder_dir}...")
                text_encoder = AutoModel.from_pretrained(text_encoder_dir)
            else:
                print(f"No text_encoder subdir; loading from: {pretrained_model_name}")
                text_encoder = AutoModel.from_pretrained(pretrained_model_name)
        except Exception as e:
            print(f"  Warning: reload ChemBERTa from name/path: {e}")
            text_encoder = AutoModel.from_pretrained(pretrained_model_name)

        model = cls(
            pretrained_model_name=pretrained_model_name,
            descriptor_dim=model_config['descriptor_dim'],
            fusion_hidden=model_config['fusion_hidden'],
            freeze_text=model_config['freeze_text'],
            use_cross_attention=model_config['use_cross_attention'],
            text_proj_dim=model_config['text_proj_dim'],
            num_classes=model_config['num_classes'],
            **kwargs
        )
        model.text_encoder = text_encoder

        weight_path = os.path.join(save_directory, 'model.safetensors')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weights: {weight_path}")

        state_dict = load_file(weight_path)
        model.load_state_dict(state_dict, strict=False)
        return model


# ----------------------------
# 4) Descriptor preprocessing
# ----------------------------
def prepare_descriptors_with_gaussianized_pca(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    smiles_col: str = "SMILES",
    n_components: Optional[int] = None,
    whiten: bool = True,
    n_quantiles: int = 1000,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], GaussianizedPCATransformer]:
    all_smiles = train_df[smiles_col].tolist()
    if val_df is not None:
        all_smiles.extend(val_df[smiles_col].tolist())
    if test_df is not None:
        all_smiles.extend(test_df[smiles_col].tolist())

    if verbose:
        print("Computing molecular descriptors...")
    desc_df, _ = compute_descriptor_matrix(all_smiles, verbose=verbose)
    desc_array = desc_df.values.astype(np.float32)

    n_train = len(train_df)
    n_val = len(val_df) if val_df is not None else 0
    n_test = len(test_df) if test_df is not None else 0

    train_desc_raw = desc_array[:n_train]
    val_desc_raw = desc_array[n_train:n_train + n_val] if n_val > 0 else None
    test_desc_raw = desc_array[n_train + n_val:n_train + n_val + n_test] if n_test > 0 else None

    if verbose:
        print(f"Fitting GaussianizedPCATransformer (n_components={n_components}, whiten={whiten})...")
    transformer = GaussianizedPCATransformer(
        n_components=n_components,
        whiten=whiten,
        n_quantiles=n_quantiles,
        random_state=random_state
    )
    train_desc = transformer.fit_transform(train_desc_raw)

    if val_desc_raw is not None:
        if verbose:
            print("Transforming validation descriptors...")
        val_desc = transformer.transform(val_desc_raw)
    else:
        val_desc = None

    if test_desc_raw is not None:
        if verbose:
            print("Transforming test descriptors...")
        test_desc = transformer.transform(test_desc_raw)
    else:
        test_desc = None

    if verbose:
        print(f"\nDescriptor preprocessing done:")
        print(f"  - Raw dim: {desc_array.shape[1]}")
        print(f"  - Transformed dim: {train_desc.shape[1]}")
        if hasattr(transformer.pca, 'explained_variance_ratio_'):
            explained_var = transformer.pca.explained_variance_ratio_
            print(f"  - Explained variance (sum): {explained_var.sum():.4f} (first 5: {explained_var[:5]})")

    return train_desc, val_desc, test_desc, transformer
