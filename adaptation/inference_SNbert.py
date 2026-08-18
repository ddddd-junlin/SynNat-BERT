import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import os
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("joblib not installed")

from utils_adapt import (
    ChemBERTaDescriptorFusion,
    MoleculeDataset,
    compute_descriptor_matrix,
    GaussianizedPCATransformer,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task',type=str)
parser.add_argument('--data',type=str)
parser.add_argument('--model',type=str,default='checkpoints/adapted_SNbert_regre')
parser.add_argument('--tokenizer',type=str,default='checkpoint/pretrained_SNbert')
parser.add_argument('--transformer',type=str, default='checkpoints/adapted/gaussianized_pca_transformer.pkl', help='Path to saved transformer (pickle file)')
args = parser.parse_args()

DATA_CSV = str(args.data)   
SMILES_COL = "SMILES"
LABEL_COL = "label"
MODEL_PATH = args.model 
CHEMBERTA_NAME = args.tokenizer
BATCH_SIZE = 128
MAX_LENGTH = 128   ##
FUSION_HIDDEN = 512   ##
num_classes = 1

PCA_N_COMPONENTS = None
PCA_WHITEN = True
PCA_N_QUANTILES = 1000

def load_transformer(transformer_path):
    if not os.path.exists(transformer_path):
        return None
    
    if HAS_JOBLIB:
        try:
            transformer = joblib.load(transformer_path)
            return transformer
        except Exception as e:
            print(f"joblib load failed")
    
    try:
        with open(transformer_path, 'rb') as f:
            transformer = pickle.load(f)
        return transformer
    except Exception as e:
        print(f"pickle load failed")
        return None

model_config_path = os.path.join(MODEL_PATH, 'model_config.json')
if os.path.exists(model_config_path):
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    EXPECTED_DESC_DIM = model_config.get('descriptor_dim', 7)
else:
    EXPECTED_DESC_DIM = 7

df = pd.read_csv(DATA_CSV)

desc_df, valid_idx = compute_descriptor_matrix(df[SMILES_COL].tolist(), verbose=True)
desc_array = desc_df.values.astype(np.float32)

transformer = None
transformer_path = args.transformer

if transformer_path:
    transformer = load_transformer(transformer_path)

if transformer is None:
    possible_names = ['transformer.pkl', 'transformer.joblib', 'gaussianized_pca_transformer.pkl']
    for name in possible_names:
        transformer_path_in_model = os.path.join(MODEL_PATH, name)
        if os.path.exists(transformer_path_in_model):
            transformer = load_transformer(transformer_path_in_model)
            if transformer is not None:
                break

# if failed refitted 
if transformer is None:
    if EXPECTED_DESC_DIM != 7:
        if PCA_N_COMPONENTS is None:
            PCA_N_COMPONENTS = EXPECTED_DESC_DIM
    
    transformer = GaussianizedPCATransformer(
        n_components=PCA_N_COMPONENTS,
        whiten=PCA_WHITEN,
        n_quantiles=PCA_N_QUANTILES,
        random_state=42
    )
    desc_array_scaled = transformer.fit_transform(desc_array)
else:
    desc_array_scaled = transformer.transform(desc_array)

if desc_array_scaled.shape[1] != EXPECTED_DESC_DIM:
    raise ValueError(f"Unmatched: expected: {EXPECTED_DESC_DIM}, actual: {desc_array_scaled.shape[1]}")
else:
    print(f"Matched: {desc_array_scaled.shape[1]}")

tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_NAME, use_fast=False)

if args.task == 'predict':
    val_dataset = MoleculeDataset(df, smiles_col=SMILES_COL, label_col=LABEL_COL,
                                  tokenizer=tokenizer, max_length=MAX_LENGTH,
                                  desc_array=desc_array_scaled)
else:
    val_dataset = MoleculeDataset(df, smiles_col=SMILES_COL,
                                  tokenizer=tokenizer, max_length=MAX_LENGTH,
                                  desc_array=desc_array_scaled)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

descriptor_dim = desc_array_scaled.shape[1]

model = ChemBERTaDescriptorFusion.from_pretrained(MODEL_PATH)

try:
    if model.use_cross_attention:
        model_desc_dim = model.cross_attn.desc_proj.in_features
    else:
        model_desc_dim = model.desc_proj[0].in_features
    
    if model_desc_dim != descriptor_dim:
        raise ValueError(f"Unmatched: expected: {EXPECTED_DESC_DIM}, actual: {desc_array_scaled.shape[1]}")
    else:
        print("Matched")
except Exception as e:
    print("Dimension unmatched")

model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        desc = batch['desc'].to(DEVICE)
        if args.task == 'predict':
            labels = batch['label'].cpu().numpy()


        result = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                desc=desc, 
                output_attentions=False, 
                return_dict=False
                )

        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result

        if isinstance(logits, torch.Tensor):
            preds = logits.view(-1).detach().cpu().numpy()
        else:
            preds = np.asarray(logits).reshape(-1)
        all_preds.extend(preds)
        if args.task == 'predict':
            all_labels.extend(labels)
        else:
            all_labels = []

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
if args.task == 'predict':
    mse = mean_squared_error(all_labels, all_preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    print(f"Validation MSE:  {mse:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation MAE:  {mae:.6f}")
    print(f"Validation R2:   {r2:.6f}")

df["prediction"] = all_preds
df.to_csv("pred_infer.csv", index=False)
print("Saved predictions to pred_infer.csv")

