"""
Parameter-Balanced Multimodal Fusion Methods for Medical Image-Report Classification
================================================================================

This module implements five standard multimodal fusion approaches with strict 
parameter balancing (~525K parameters each) for fair comparison:

1. EarlyConcat: Simple concatenation of unimodal features
2. GMU: Gated Multimodal Unit (Arevalo et al., 2017)
3. BilinearFusion: Element-wise product of projected features
4. JointTransformer: Self-attention over concatenated tokens (Vaswani et al., 2017)
5. LowRankFusion: Low-rank tensor decomposition (Liu et al., 2018)

"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import time
import json
import warnings
import argparse
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for multimodal fusion experiments."""
    img_dim: int = 1024          # DenseNet-121 output dimension
    txt_dim: int = 768           # BioClinicalBERT output dimension
    fusion_dim: int = 256        # Target fusion dimension
    low_rank: int = 32           # Rank for LowRankFusion
    batch_size: int = 16
    epochs: int = 20            
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5
    dropout: float = 0.3
    max_len: int = 256
    random_state: int = 42
    n_bootstrap: int = 1000

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default paths (can be overridden via command line)
DATA_PATHS = {
    "proj_csv": "/kaggle/input/datasets/raddar/chest-xrays-indiana-university/indiana_projections.csv",
    "report_csv": "/kaggle/input/datasets/raddar/chest-xrays-indiana-university/indiana_reports.csv",
    "image_dir": "/kaggle/input/datasets/raddar/chest-xrays-indiana-university/images/images_normalized",
    "output_dir": "/kaggle/working/fusion_results",
    "models_dir": "/kaggle/working/fusion_results/models",
    "plots_dir": "/kaggle/working/fusion_results/plots"
}

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# DATASET
# ============================================================================

class IndianaDataset(Dataset):
    """Dataset for Indiana University chest X-rays with reports."""
    
    def __init__(self, proj_csv: str, report_csv: str, image_dir: str, 
                 tokenizer, transform):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load and merge data
        df_proj = pd.read_csv(proj_csv)
        df_rep = pd.read_csv(report_csv)
        df = df_proj.merge(df_rep, on="uid", how="inner")
        df = df[df["projection"] == "Frontal"].reset_index(drop=True)
        
        # Filter missing images
        valid_rows = []
        for idx, row in df.iterrows():
            if os.path.exists(os.path.join(self.image_dir, row["filename"])):
                valid_rows.append(row)
            else:
                print(f"Warning: Missing image {row['filename']} - skipping.")
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        
        # Prepare text and labels
        self.df["text"] = self.df["findings"].fillna(self.df["impression"]).fillna("normal")
        
        abnormal_keywords = ["opacity", "effusion", "pneumonia", "cardiomegaly", 
                          "edema", "consolidation", "mass", "nodule"]
        self.df["label"] = self.df["text"].str.lower().apply(
            lambda x: 1 if any(k in x for k in abnormal_keywords) else 0
        )
        
        # Compute class weights for imbalance
        n_total = len(self.df)
        n_abnormal = self.df["label"].sum()
        n_normal = n_total - n_abnormal
        self.class_weights = torch.tensor([
            n_total / (2 * n_normal) if n_normal > 0 else 1.0,
            n_total / (2 * n_abnormal) if n_abnormal > 0 else 1.0
        ], dtype=torch.float32)
        
        print(f"Dataset: {n_total} samples, {n_abnormal} abnormal ({100*n_abnormal/n_total:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Tokenize text
        enc = self.tokenizer(
            row["text"], 
            max_length=CFG.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.float32)
        }

# ============================================================================
# ENCODERS (Frozen Backbone)
# ============================================================================

class ImageEncoder(nn.Module):
    """DenseNet-121 backbone for image encoding (frozen)."""
    
    def __init__(self):
        super().__init__()
        backbone = models.densenet121(weights="IMAGENET1K_V1")
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim = CFG.img_dim
        
        # FIX: Disable inplace ReLU for CUDA compatibility
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        return feat.view(feat.size(0), -1)


class TextEncoder(nn.Module):
    """BioClinicalBERT backbone for text encoding (frozen)."""
    
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.dim = CFG.txt_dim
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]

# ============================================================================
# SHARED CLASSIFIER
# ============================================================================

class UnifiedClassifier(nn.Module):
    """
    Shared classification head.
    Parameters: (256*256 + 256) + (256 + 1) = 66,305
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),  # No inplace for CUDA safety
            nn.Dropout(CFG.dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)

# ============================================================================
# FUSION METHODS (~525K parameters each)
# ============================================================================

class EarlyConcat(nn.Module):
    """
    Early concatenation fusion baseline.
    Simply concatenates image and text features.
    """
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(CFG.img_dim + CFG.txt_dim, CFG.fusion_dim),
            nn.ReLU(),  # No inplace
            nn.Dropout(CFG.dropout)
        )
        self.classifier = UnifiedClassifier()
        
        print(f"[EarlyConcat] {count_parameters(self):,} parameters")
    
    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([img, txt], dim=1))
        return self.classifier(fused)


class GMU(nn.Module):
    """
    Gated Multimodal Unit (Arevalo et al., 2017).
    
    Original formulation:
    h_v = tanh(W_v · x_v + b_v)
    h_t = tanh(W_t · x_t + b_t)
    z = σ(W_z · [h_v; h_t] + b_z)  # Gate from TRANSFORMED features
    h = z ⊙ h_v + (1-z) ⊙ h_t
    
    Reference: https://doi.org/10.1109/ICCV.2017.172
    """
    def __init__(self):
        super().__init__()
        bottleneck = 188  # Bottleneck dimension
        
        # Modality-specific transformations
        self.img_proj = nn.Linear(CFG.img_dim, bottleneck)
        self.txt_proj = nn.Linear(CFG.txt_dim, bottleneck)
        
        # CORRECTED: Gate computed from TRANSFORMED features (not raw)
        self.gate = nn.Linear(bottleneck * 2, bottleneck)
        
        # Upscale to fusion dimension
        self.upscale = nn.Sequential(
            nn.Linear(bottleneck, CFG.fusion_dim),
            nn.ReLU(),  # No inplace
            nn.Dropout(CFG.dropout)
        )
        self.classifier = UnifiedClassifier()
        
        print(f"[GMU] {count_parameters(self):,} parameters (bottleneck={bottleneck})")
    
    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        # Transform modalities
        h_img = torch.tanh(self.img_proj(img))
        h_txt = torch.tanh(self.txt_proj(txt))
        
        # CORRECTED: Gate from concatenated TRANSFORMED features
        gate = torch.sigmoid(self.gate(torch.cat([h_img, h_txt], dim=1)))
        
        # Gated fusion
        fused = gate * h_img + (1 - gate) * h_txt
        fused = self.upscale(fused)
        return self.classifier(fused)


class BilinearFusion(nn.Module):
    """
    Bilinear fusion via element-wise product.
    Projects modalities to common space then combines multiplicatively.
    """
    def __init__(self):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(CFG.img_dim, CFG.fusion_dim),
            nn.ReLU()  # No inplace
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(CFG.txt_dim, CFG.fusion_dim),
            nn.ReLU()  # No inplace
        )
        self.dropout = nn.Dropout(CFG.dropout)
        self.classifier = UnifiedClassifier()
        
        print(f"[BilinearFusion] {count_parameters(self):,} parameters")
    
    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        img_h = self.img_proj(img)
        txt_h = self.txt_proj(txt)
        fused = self.dropout(img_h * txt_h)  # Element-wise product
        return self.classifier(fused)


class JointTransformer(nn.Module):
    """
    Joint Transformer fusion (Vaswani et al., 2017).
    
    CORRECTED: Full Transformer block with:
    1. Multi-head self-attention
    2. Add & LayerNorm
    3. Feed-Forward Network
    4. Add & LayerNorm
    
    Reference: https://arxiv.org/abs/1706.03762
    """
    def __init__(self):
        super().__init__()
        # Reduced dimension to fit parameter budget
        self.attn_dim = 128
        
        self.img_proj = nn.Linear(CFG.img_dim, self.attn_dim)
        self.txt_proj = nn.Linear(CFG.txt_dim, self.attn_dim)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            self.attn_dim, 
            num_heads=4, 
            dropout=CFG.dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.attn_dim)
        
        # CORRECTED: Feed-Forward Network (required for Transformer block)
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim * 4),
            nn.ReLU(),  # No inplace
            nn.Dropout(CFG.dropout),
            nn.Linear(self.attn_dim * 4, self.attn_dim)
        )
        self.norm2 = nn.LayerNorm(self.attn_dim)
        self.dropout = nn.Dropout(CFG.dropout)
        
        # Output projection
        self.output_proj = nn.Linear(self.attn_dim, CFG.fusion_dim)
        self.classifier = UnifiedClassifier()
        
        print(f"[JointTransformer] {count_parameters(self):,} parameters (dim={self.attn_dim})")
    
    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        # Project to tokens
        img_tok = self.img_proj(img).unsqueeze(1)  # (batch, 1, dim)
        txt_tok = self.txt_proj(txt).unsqueeze(1)   # (batch, 1, dim)
        x = torch.cat([img_tok, txt_tok], dim=1)    # (batch, 2, dim)
        
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # CORRECTED: FFN with residual (standard Transformer block)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Global pooling
        fused = self.dropout(x.mean(dim=1))
        fused = self.output_proj(fused)
        return self.classifier(fused)


class LowRankFusion(nn.Module):
    """
    Low-Rank Multimodal Fusion (Liu et al., 2018).
    
    Decomposes high-dimensional tensor product using low-rank factors:
    h = sum_{i=1}^k (U_img[:,i] ⊙ U_txt[:,i]) where ⊙ is element-wise product
    
    Reference: https://aclanthology.org/P18-1209/
    """
    def __init__(self):
        super().__init__()
        self.rank = CFG.low_rank  # k = 32
        
        # Project to common dimension
        self.img_proj = nn.Linear(CFG.img_dim, CFG.fusion_dim)
        self.txt_proj = nn.Linear(CFG.txt_dim, CFG.fusion_dim)
        
        # Low-rank factor matrices
        self.U_img = nn.Parameter(torch.randn(CFG.fusion_dim, self.rank))
        self.U_txt = nn.Parameter(torch.randn(CFG.fusion_dim, self.rank))
        
        # Output transformation
        self.W_out = nn.Linear(self.rank, CFG.fusion_dim)
        
        nn.init.xavier_uniform_(self.U_img)
        nn.init.xavier_uniform_(self.U_txt)
        
        self.dropout = nn.Dropout(CFG.dropout)
        self.classifier = UnifiedClassifier()
        
        print(f"[LowRankFusion] {count_parameters(self):,} parameters (rank={self.rank})")
    
    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        # Project to common space
        img_h = self.img_proj(img)  # (batch, fusion_dim)
        txt_h = self.txt_proj(txt)   # (batch, fusion_dim)
        
        # Low-rank interaction: compute rank components
        img_comp = torch.matmul(img_h, self.U_img)  # (batch, rank)
        txt_comp = torch.matmul(txt_h, self.U_txt)   # (batch, rank)
        
        # Element-wise product across rank
        interaction = img_comp * txt_comp  # (batch, rank)
        interaction = self.dropout(interaction)
        
        # Project to fusion dimension
        fused = self.W_out(interaction)  # (batch, fusion_dim)
        return self.classifier(fused)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model: nn.Module, img_enc: nn.Module, txt_enc: nn.Module,
                loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        optimizer.zero_grad()
        
        images = batch["image"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        with torch.no_grad():
            img_feat = img_enc(images)
            txt_feat = txt_enc(input_ids, attention_mask)
        
        logits = model(img_feat, txt_feat)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, img_enc: nn.Module, txt_enc: nn.Module,
             loader: DataLoader) -> Dict[str, Any]:
    """Evaluate model on a dataset."""
    model.eval()
    all_probs, all_labels = [], []
    
    for batch in loader:
        images = batch["image"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        
        img_feat = img_enc(images)
        txt_feat = txt_enc(input_ids, attention_mask)
        logits = model(img_feat, txt_feat)
        probs = torch.sigmoid(logits)
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    }


def bootstrap_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                      n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for metrics."""
    rng = np.random.RandomState(42)
    metrics = {"auc": [], "f1": [], "accuracy": [], "precision": [], "recall": []}
    
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        
        y_t, y_p = y_true[idx], y_prob[idx]
        y_pr = (y_p > 0.5).astype(int)
        
        metrics["auc"].append(roc_auc_score(y_t, y_p))
        metrics["f1"].append(f1_score(y_t, y_pr, zero_division=0))
        metrics["accuracy"].append(accuracy_score(y_t, y_pr))
        metrics["precision"].append(precision_score(y_t, y_pr, zero_division=0))
        metrics["recall"].append(recall_score(y_t, y_pr, zero_division=0))
    
    results = {}
    for k, v in metrics.items():
        arr = np.array(v)
        results[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95": (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        }
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history: Dict, model_name: str, plots_dir: str) -> None:
    """Plot training curves for a model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0].plot(epochs, history["train_loss"], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_title(f'{model_name} - Training Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history["val_auc"], 'r-', linewidth=2, label='Val AUC')
    best_epoch = np.argmax(history["val_auc"])
    axes[1].axvline(x=best_epoch + 1, color='g', linestyle='--', 
                   label=f'Best: {max(history["val_auc"]):.4f} (Epoch {best_epoch+1})')
    axes[1].set_title(f'{model_name} - Validation AUC', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history["val_acc"], 'g-', linewidth=2, label='Accuracy')
    axes[2].plot(epochs, history["val_f1"], 'orange', linewidth=2, label='F1-score')
    axes[2].set_title(f'{model_name} - Validation Metrics', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(results: Dict, plots_dir: str) -> None:
    """Plot comparison of all models."""
    models = list(results.keys())
    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = custom_colors[:len(models)]
    
    fig = plt.figure(figsize=(18, 8))
    
    for idx, metric in enumerate(metrics):
        ax = plt.subplot(2, 3, idx + 1)
        values = [results[m]["metrics"][metric]["mean"] for m in models]
        errors = [results[m]["metrics"][metric]["std"] for m in models]
        
        bars = ax.bar(models, values, yerr=errors, capsize=5, 
                     alpha=0.85, edgecolor='black', linewidth=1.2, color=colors)
        ax.set_title(f'{metric.upper()}', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric.upper())
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.5, 1.0])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Radar chart (linear scale - FIXED)
    ax = plt.subplot(2, 3, 6, projection='polar')
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for model_idx, model in enumerate(models):
        values = [results[model]["metrics"][m]["mean"] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=3, label=model, 
               color=colors[model_idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[model_idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
    ax.set_title('Radar Chart', fontweight='bold', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True)
    ax.set_ylim([0.5, 1.0])
    
    plt.suptitle('Multimodal Fusion Strategies Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    fig_log, ax_log = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model_idx, model in enumerate(models):
        values = [results[model]["metrics"][m]["mean"] for m in metrics]
        values_log = [np.log10(v) for v in values]
        values_log += values_log[:1]
        ax_log.plot(angles, values_log, 'o-', linewidth=3, label=model, 
                   color=colors[model_idx], markersize=8)
        ax_log.fill(angles, values_log, alpha=0.15, color=colors[model_idx])
    
    original_values =[0.80, 0.85, 0.90, 0.95, 1.00]
    log_positions = [np.log10(v) for v in original_values]
    
    ax_log.set_yticks(log_positions)
    ax_log.set_yticklabels([f'{v:.2f}' for v in original_values], fontsize=10)
    
    ax_log.set_ylim([np.log10(0.8), 0])  
    
    ax_log.set_xticks(angles[:-1])
    ax_log.set_xticklabels([m.upper() for m in metrics], fontsize=12, fontweight='bold')
    ax_log.set_title('Radar Logarithmic Scale', fontweight='bold', fontsize=16, pad=30)
    ax_log.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
    ax_log.grid(True)
    
    ax_log.set_rgrids(log_positions, labels=[f'{v:.2f}' for v in original_values])
    
    plt.tight_layout()
    log_save_path = os.path.join(plots_dir, 'model_comparison_radar_log.png')
    plt.savefig(log_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig_log)
    
    print(f" Saved radar charts to {plots_dir}")


def plot_confusion_matrices(all_results: Dict, plots_dir: str) -> None:
    """Plot confusion matrices for all models."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        cm = confusion_matrix(results["raw"]["y_true"], results["raw"]["y_pred"])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=['Normal', 'Abnormal'], 
                   yticklabels=['Normal', 'Abnormal'],
                   ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{model_name}Accuracy: {results["metrics"]["accuracy"]["mean"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.suptitle('Confusion Matrices (Normalized)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main experiment pipeline."""
    print(f"{'='*80}")
    print("MULTIMODAL FUSION COMPARISON")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    
    seed_everything(42)
    
    # Create directories
    os.makedirs(DATA_PATHS["output_dir"], exist_ok=True)
    os.makedirs(DATA_PATHS["models_dir"], exist_ok=True)
    os.makedirs(DATA_PATHS["plots_dir"], exist_ok=True)
    
    # Load data
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = IndianaDataset(
        DATA_PATHS["proj_csv"],
        DATA_PATHS["report_csv"],
        DATA_PATHS["image_dir"],
        tokenizer,
        transform
    )
    
    # Train/val/test split
    labels = [dataset.df.iloc[i]["label"] for i in range(len(dataset))]
    train_val_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.15, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.176, stratify=train_val_labels, random_state=42
    )
    
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Reproducible DataLoaders
    g = torch.Generator()
    g.manual_seed(42)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx), 
        batch_size=CFG.batch_size, 
        shuffle=True, 
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), 
        batch_size=CFG.batch_size, 
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), 
        batch_size=CFG.batch_size, 
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Initialize encoders
    img_enc = ImageEncoder().to(DEVICE).eval()
    txt_enc = TextEncoder().to(DEVICE).eval()
    
    # Initialize fusion models
    fusion_models = {
        "EarlyConcat": EarlyConcat(),
        "GMU": GMU(),
        "BilinearFusion": BilinearFusion(),
        "JointTransformer": JointTransformer(),
        "LowRankFusion": LowRankFusion()
    }
    
    # Verify parameter balance
    print(f"{'='*80}")
    print("PARAMETER COUNT VERIFICATION")
    print(f"{'='*80}")
    params_list = []
    for name, model in fusion_models.items():
        p = count_parameters(model)
        params_list.append((name, p))
        print(f"{name:<20}: {p:,} parameters")
    
    min_p = min(p for _, p in params_list)
    max_p = max(p for _, p in params_list)
    ratio = max_p / min_p
    print(f"Parameter ratio (max/min): {ratio:.2f}x")
    if ratio > 1.5:
        print("WARNING: Parameter imbalance >1.5x may affect fairness!")
    else:
        print("Parameter balance acceptable for fair comparison")
    print(f"{'='*80}")
    
    # Train each model
    results = {}
    all_histories = {}
    
    for name, model in fusion_models.items():
        print(f"{'='*80}")
        print(f"TRAINING: {name}")
        print(f"{'='*80}")
        
        model = model.to(DEVICE)
        params = count_parameters(model)
        
        # Weighted loss for class imbalance
        pos_weight = dataset.class_weights[1] / dataset.class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        
        early_stop = EarlyStopping(patience=CFG.patience)
        best_state = None
        best_auc = 0
        history = {"train_loss": [], "val_auc": [], "val_acc": [], 
                   "val_f1": [], "val_precision": [], "val_recall": []}
        
        start_time = time.time()
        
        for epoch in range(CFG.epochs):
            loss = train_epoch(model, img_enc, txt_enc, train_loader, optimizer, criterion)
            val_m = evaluate(model, img_enc, txt_enc, val_loader)
            
            history["train_loss"].append(loss)
            history["val_auc"].append(val_m["auc"])
            history["val_acc"].append(val_m["accuracy"])
            history["val_f1"].append(val_m["f1"])
            
            if val_m["auc"] > best_auc:
                best_auc = val_m["auc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if early_stop(val_m["auc"]):
                print(f"Early stop at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Val AUC: {val_m['auc']:.4f}")
        
        training_time = time.time() - start_time
        all_histories[name] = history.copy()
        
        # Plot training curves
        plot_training_curves(history, name, DATA_PATHS["plots_dir"])
        
        # Test evaluation
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        test_m = evaluate(model, img_enc, txt_enc, test_loader)
        
        print(f"Computing bootstrap CIs...")
        ci = bootstrap_metrics(test_m["y_true"], test_m["y_prob"], CFG.n_bootstrap)
        
        results[name] = {
            "params": params,
            "training_time": training_time,
            "metrics": ci,
            "raw": {
                "y_true": test_m["y_true"].tolist(),
                "y_prob": test_m["y_prob"].tolist(),
                "y_pred": test_m["y_pred"].tolist()
            }
        }
        
        print(f">>> TEST RESULTS:")
        print(f"AUC:  {ci['auc']['mean']:.4f} [{ci['auc']['ci_95'][0]:.4f}-{ci['auc']['ci_95'][1]:.4f}]")
        print(f"F1:   {ci['f1']['mean']:.4f} [{ci['f1']['ci_95'][0]:.4f}-{ci['f1']['ci_95'][1]:.4f}]")
        
        # Save model
        model_save_path = os.path.join(DATA_PATHS["models_dir"], f"{name}_best.pt")
        torch.save(best_state, model_save_path)
    
    # Final summary
    print(f"{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Params':<12} {'AUC':<25} {'F1':<25}")
    print("-" * 85)
    
    for name in fusion_models.keys():
        r = results[name]
        m = r["metrics"]
        print(f"{name:<20} {r['params']:<12,} "
              f"{m['auc']['mean']:.3f} [{m['auc']['ci_95'][0]:.3f}-{m['auc']['ci_95'][1]:.3f}]   "
              f"{m['f1']['mean']:.3f} [{m['f1']['ci_95'][0]:.3f}-{m['f1']['ci_95'][1]:.3f}]")
    
    print(f"{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}")
    
    plot_confusion_matrices(results, DATA_PATHS["plots_dir"])
    plot_comparison(results, DATA_PATHS["plots_dir"])
    
    # Save results
    results_path = os.path.join(DATA_PATHS["output_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Plots saved to: {DATA_PATHS['plots_dir']}")
    print(f"Models saved to: {DATA_PATHS['models_dir']}")
    print(f"Results saved to: {results_path}")
    
    print(f"{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Medical Fusion - Publication Ready")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--proj_csv", type=str, default=DATA_PATHS["proj_csv"])
    parser.add_argument("--report_csv", type=str, default=DATA_PATHS["report_csv"])
    parser.add_argument("--image_dir", type=str, default=DATA_PATHS["image_dir"])
    parser.add_argument("--output_dir", type=str, default=DATA_PATHS["output_dir"])
    
    args, unknown = parser.parse_known_args()
    
    CFG.batch_size = args.batch_size
    CFG.epochs = args.epochs
    
    DATA_PATHS["proj_csv"] = args.proj_csv
    DATA_PATHS["report_csv"] = args.report_csv
    DATA_PATHS["image_dir"] = args.image_dir
    DATA_PATHS["output_dir"] = args.output_dir
    DATA_PATHS["models_dir"] = os.path.join(args.output_dir, "models")
    DATA_PATHS["plots_dir"] = os.path.join(args.output_dir, "plots")
    
    main()