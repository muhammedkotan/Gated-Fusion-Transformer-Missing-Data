"""
models.py
=========
All neural-network architectures for the Gated Fusion project.
"""

import math
import torch
import torch.nn as nn

from config import N_SALES_FEATURES, N_VIEW_FEATURES, N_TOTAL_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# 1. BASELINE TRANSFORMER
# ═════════════════════════════════════════════════════════════════════════════
class BaselineTransformer(nn.Module):
    """
    Single-stream Transformer.  
    """

    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2,
                 n_categories: int = 18, dropout: float = 0.1):
        super().__init__()
        self.category_embedding  = nn.Embedding(n_categories, 8)
        self.input_projection    = nn.Linear(input_dim, d_model)
        self.category_projection = nn.Linear(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1), nn.ReLU())

    def forward(self, x: torch.Tensor, categories: torch.Tensor,
                view_mask=None) -> torch.Tensor:
        x_proj   = self.input_projection(x)
        cat_emb  = self.category_embedding(categories)
        cat_proj = self.category_projection(cat_emb)
        seq      = torch.stack([x_proj, cat_proj], dim=1)
        out      = self.transformer(seq)
        return self.output(out[:, 0, :]).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DUAL-STREAM TRANSFORMER
# ═════════════════════════════════════════════════════════════════════════════
class DualStreamTransformer(nn.Module):
    """
    Two parallel projection streams (sales / engagement) combined with a
    fixed weight of 0.5 on the view stream when engagement data is available.
    """

    def __init__(self, sales_dim: int, view_dim: int, d_model: int = 32,
                 nhead: int = 2, n_categories: int = 18, dropout: float = 0.1):
        super().__init__()
        self.category_embedding = nn.Embedding(n_categories, 8)
        self.sales_projection   = nn.Linear(sales_dim + 8, d_model)
        self.view_projection    = nn.Linear(view_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1), nn.ReLU())

    def forward(self, x_sales: torch.Tensor, x_view: torch.Tensor,
                categories: torch.Tensor,
                view_mask: torch.Tensor) -> torch.Tensor:
        cat_emb         = self.category_embedding(categories)
        sales_with_cat  = torch.cat([x_sales, cat_emb], dim=1)
        sales_proj      = self.sales_projection(sales_with_cat)
        view_proj       = self.view_projection(x_view)
        view_mask_exp   = view_mask.unsqueeze(1).float()
        combined        = sales_proj + 0.5 * view_proj * view_mask_exp
        seq             = combined.unsqueeze(1)
        out             = self.transformer(seq)
        return self.output(out.squeeze(1)).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 3. ADAPTIVE WEIGHTING TRANSFORMER
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveWeightingTransformer(nn.Module):
    """
    Scalar alpha weight: complete rows receive weight 1.0, missing rows
    receive weight alpha (learned via HPO).
    """

    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2,
                 n_categories: int = 18, dropout: float = 0.1,
                 alpha: float = 0.5):
        super().__init__()
        self.alpha               = alpha
        self.category_embedding  = nn.Embedding(n_categories, 8)
        self.feature_projection  = nn.Linear(input_dim, d_model)
        self.category_projection = nn.Linear(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1), nn.ReLU())

    def forward(self, x: torch.Tensor, categories: torch.Tensor,
                view_mask: torch.Tensor) -> torch.Tensor:
        feat_proj = self.feature_projection(x)
        mask_exp  = view_mask.unsqueeze(1).float()
        weighted  = feat_proj * (mask_exp + (1 - mask_exp) * self.alpha)
        cat_emb   = self.category_embedding(categories)
        cat_proj  = self.category_projection(cat_emb)
        seq       = torch.stack([weighted, cat_proj], dim=1)
        out       = self.transformer(seq)
        return self.output(out[:, 0, :]).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 4. GATED FUSION TRANSFORMER  ← paper's proposed model
# ═════════════════════════════════════════════════════════════════════════════
class GatedFusionTransformer(nn.Module):
    """
    Proposed model: two independent Transformer streams (sales-only and
    full features) whose predictions are combined via a *learned soft gate*.

    Gate inputs: category embedding + binary view-availability flag.
    Gate outputs: (g_sales, g_full) with g_sales + g_full = 1 via Softmax.
    Final prediction: ŷ = g_sales · ŷ_sales + g_full · ŷ_full

    The gate adapts at inference time: when engagement data is missing the
    gate up-weights the sales stream; when it is present the gate can
    leverage the full-feature stream — without any hard-coded rule.
    """

    def __init__(self, sales_dim: int, full_dim: int, d_model: int = 32,
                 nhead: int = 2, num_layers: int = 1, dropout: float = 0.1,
                 n_categories: int = 18, category_emb_dim: int = 8):
        super().__init__()
        # Shared category embedding
        self.category_embedding = nn.Embedding(n_categories, category_emb_dim)

        # Sales-only stream
        self.sales_projection = nn.Linear(sales_dim + category_emb_dim, d_model)
        sales_enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.sales_transformer = nn.TransformerEncoder(sales_enc, num_layers)
        self.sales_output = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1), nn.ReLU())

        # Full-feature stream
        self.full_projection = nn.Linear(full_dim + category_emb_dim, d_model)
        full_enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.full_transformer = nn.TransformerEncoder(full_enc, num_layers)
        self.full_output = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1), nn.ReLU())

        # Learned gate: [category_emb_dim + 1] → 2 Softmax weights
        self.gate = nn.Sequential(
            nn.Linear(category_emb_dim + 1, 16), nn.ReLU(),
            nn.Linear(16, 2), nn.Softmax(dim=1))

    def forward(self, x_sales: torch.Tensor, x_all: torch.Tensor,
                categories: torch.Tensor,
                view_mask: torch.Tensor) -> torch.Tensor:
        cat_emb = self.category_embedding(categories)

        # Sales stream
        sales_in   = torch.cat([x_sales, cat_emb], dim=1)
        sales_proj = self.sales_projection(sales_in).unsqueeze(1)
        sales_out  = self.sales_transformer(sales_proj)
        sales_pred = self.sales_output(sales_out.squeeze(1))

        # Full stream
        full_in   = torch.cat([x_all, cat_emb], dim=1)
        full_proj = self.full_projection(full_in).unsqueeze(1)
        full_out  = self.full_transformer(full_proj)
        full_pred = self.full_output(full_out.squeeze(1))

        # Gate
        view_flag    = view_mask.float().unsqueeze(1)
        gate_input   = torch.cat([cat_emb, view_flag], dim=1)
        gate_weights = self.gate(gate_input)

        final_pred = (gate_weights[:, 0:1] * sales_pred
                      + gate_weights[:, 1:2] * full_pred)
        return final_pred.squeeze(-1)


ConfigurableEnsembleTransformer = GatedFusionTransformer


# ═════════════════════════════════════════════════════════════════════════════
# 5. SAINT MODEL  (Self-Attention and Intersample Attention Transformer)
# Architecture: feature tokenisation → SAINT blocks (column + row attention)
# → MLP regression head.
# ═════════════════════════════════════════════════════════════════════════════

class _MultiHeadAttention(nn.Module):
    """Lightweight multi-head self-attention used inside SAINTBlock."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead   = nhead
        self.d_head  = d_model // nhead
        self.scale   = math.sqrt(self.d_head)
        self.q       = nn.Linear(d_model, d_model)
        self.k       = nn.Linear(d_model, d_model)
        self.v       = nn.Linear(d_model, d_model)
        self.out     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q(x).reshape(B, N, self.nhead, self.d_head).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.nhead, self.d_head).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.nhead, self.d_head).transpose(1, 2)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class SAINTBlock(nn.Module):
    """
    One SAINT block: column (feature-wise) attention then row
    (intersample) attention.
    """

    def __init__(self, n_features: int, d_model: int, nhead: int,
                 dropout: float = 0.1):
        super().__init__()
        # Column attention
        self.col_norm1 = nn.LayerNorm(d_model)
        self.col_attn  = _MultiHeadAttention(d_model, nhead, dropout)
        self.col_norm2 = nn.LayerNorm(d_model)
        self.col_ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

        # Row attention — operates on flattened feature dimension
        row_dim          = n_features * d_model
        self.row_norm1   = nn.LayerNorm(row_dim)
        self.row_attn    = nn.MultiheadAttention(
            row_dim, 1, dropout=dropout, batch_first=True)
        self.row_norm2   = nn.LayerNorm(row_dim)
        self.row_ff      = nn.Sequential(
            nn.Linear(row_dim, row_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(row_dim * 2, row_dim))

        self.n_features = n_features
        self.d_model    = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_features, d_model]
        # Column attention
        x = x + self.col_attn(self.col_norm1(x))
        x = x + self.col_ff(self.col_norm2(x))

        # Row attention — flatten across features
        B, N, D  = x.shape
        x_flat   = x.reshape(B, N * D).unsqueeze(1)    # [B, 1, N*D]
        x_norm   = self.row_norm1(x_flat)
        row_out, _ = self.row_attn(x_norm, x_norm, x_norm)
        x_flat   = x_flat + row_out
        x_flat   = x_flat + self.row_ff(self.row_norm2(x_flat))
        x        = x_flat.squeeze(1).reshape(B, N, D)
        return x


class SAINTModel(nn.Module):
    """
    Full-SAINT (colrow) for regression on tabular data with systematic
    missingness.  
    """

    def __init__(self, n_cont_features: int = N_TOTAL_FEATURES,
                 n_categories: int = 18, d_model: int = 32,
                 nhead: int = 2, n_layers: int = 2,
                 dropout: float = 0.1, category_emb_dim: int = 8):
        super().__init__()
        self.n_cont  = n_cont_features
        self.d_model = d_model

        # Per-feature linear embeddings
        self.feat_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_cont_features)])

        # Category token embedding
        self.cat_embedding = nn.Embedding(n_categories, d_model)

        # Total tokens = continuous features + 1 category token
        self.n_tokens = n_cont_features + 1

        # SAINT blocks
        self.blocks = nn.ModuleList([
            SAINTBlock(self.n_tokens, d_model, nhead, dropout)
            for _ in range(n_layers)])

        # Regression head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * self.n_tokens, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU())

    def forward(self, x_cont: torch.Tensor, categories: torch.Tensor,
                view_mask=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x_cont     : (B, n_cont_features) — zero-filled features
        categories : (B,)                 — integer category indices
        view_mask  : ignored (API compatibility with other models)
        """
        # Tokenise each continuous feature
        tokens = [emb(x_cont[:, i:i+1])
                  for i, emb in enumerate(self.feat_embeddings)]
        tokens.append(self.cat_embedding(categories))       # category token
        x = torch.stack(tokens, dim=1)                     # [B, n_tokens, D]

        # SAINT blocks
        for block in self.blocks:
            x = block(x)

        # Aggregate and regress
        x = self.norm(x).reshape(x.shape[0], -1)           # [B, n_tokens*D]
        return self.head(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
