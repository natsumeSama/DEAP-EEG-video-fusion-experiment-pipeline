from typing import Optional

import torch
from torch import nn

"""
Model zoo / factory for DEAP EEG experiments.

This module groups baseline models that match the 4 dataset "modes" produced
by the data pipeline.

Modes
-----
mode 0/1:
    - Time-domain EEG (raw or filtered), shape (B, C=32, T)
    - Models: 1D CNN baselines

mode 2:
    - Per-trial sequence of window features, shape (B, Nwin, F)
    - Models: temporal Conv1D baseline or Transformer encoder baseline

mode 3:
    - Per-window features flattened across all trials, shape (B, F)
    - Model: small MLP classifier

The `build_model(...)` factory returns the correct model for the selected mode.
"""


# =========================
# Mode 0/1: time-domain CNN
# =========================
class CNNTimeDomain(nn.Module):
    """
    Time-domain CNN baseline for modes 0 and 1.

    Expected input:
        x: (B, C=32, T)
           - B: batch size
           - C: number of EEG channels
           - T: number of time samples in one trial

    Output:
        logits: (B, num_classes)

    Notes:
        This model applies 1D convolutions along the time axis.
        It is a compact baseline that is useful for:
        - fast experiments
        - sanity-checking the pipeline
        - comparing raw vs filtered EEG
    """
    def __init__(self, num_channels: int = 32, num_classes: int = 2):
        super().__init__()

        # First block:
        # extract early temporal patterns and reduce sequence length
        self.convblock1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
        )

        # Second block:
        # learn deeper temporal representations
        self.convblock2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        # Third block:
        # refine learned features while keeping the same channel width
        self.convblock3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        # Classification head:
        # pool over time, flatten, regularize, then classify
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x: EEG batch of shape (B, C, T)

        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return self.head(x)


class TripathiSmallCNN1D(nn.Module):
    """
    Small and shallow 1D CNN baseline for DEAP trials (modes 0/1).

    Expected input:
        x: (B, 32, T)

    Output:
        logits: (B, num_classes)

    Why this model:
        This is a lightweight baseline inspired by shallow CNN setups used in
        older EEG emotion recognition work. It is simple, stable, and fast to train.
    """
    def __init__(self, num_channels=32, num_classes=2, p_drop=0.5):
        super().__init__()

        self.emb_dim = 128

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )

        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Linear(self.emb_dim, num_classes)

    def encode(self, x):
        return self.features(x)   # (B, 128)

    def forward(self, x):
        z = self.encode(x)
        z = self.dropout(z)
        return self.head(z)


# ===========================================
# Mode 2: per-trial SEQUENCE model over DE/PSD
# ===========================================
class SeqConv1D(nn.Module):
    """
    Temporal Conv1D baseline for feature sequences (mode 2).

    Expected input:
        x: (B, Nwin, F)
           - Nwin: number of windows per trial
           - F: feature dimension for each window

    Output:
        logits: (B, num_classes)

    Pipeline:
        1) Project each window feature vector from F to hidden_dim
        2) Transpose to (B, hidden_dim, Nwin) so Conv1D runs over windows
        3) Apply temporal Conv1D layers
        4) Pool across windows and classify
    """
    def __init__(
        self,
        feature_dim: int,         # F
        num_classes: int = 2,
        hidden_dim: int = 128,    # D
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project each window independently into a hidden representation
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Temporal Conv1D stack over the window axis
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Pool sequence information into one vector, then classify
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x: Feature sequence batch of shape (B, Nwin, F)

        Returns:
            Logits of shape (B, num_classes)
        """
        # x: (B, Nwin, F)
        B, N, F = x.shape

        # Step 1: project each window feature vector
        x = self.proj(x)          # (B, N, D)

        # Step 2: rearrange dimensions for Conv1D
        x = x.transpose(1, 2)     # (B, D, N)

        # Step 3: temporal modeling across windows
        x = self.temporal(x)      # (B, D, N)

        # Step 4: global pooling + classification
        return self.head(x)       # (B, num_classes)


class EEGFeatureTransformer(nn.Module):
    """
    Transformer encoder baseline for EEG feature sequences (mode 2).

    Expected input:
        x: (B, Nwin, F)

    Output:
        logits: (B, num_classes)

    Important:
        - `encode(x)` returns the sequence embedding before classification
        - `forward(x)` returns normal classification logits
    """
    def __init__(
        self,
        feat_dim,
        num_classes=2,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Project each window feature vector into the transformer embedding space
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Standard Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier used in EEG-only training
        self.head = nn.Linear(d_model, num_classes)

    def encode(self, x):
        """
        Encode the input sequence into one EEG embedding.

        Args:
            x: Input tensor of shape (B, Nwin, F)

        Returns:
            z: Embedding tensor of shape (B, d_model)
        """
        x = self.input_proj(x)   # (B, Nwin, D)
        x = self.encoder(x)      # (B, Nwin, D)

        # Mean pooling across windows to get one vector per sample
        z = x.mean(dim=1)        # (B, D)
        return z

    def forward(self, x):
        """
        Standard classification forward pass.

        Args:
            x: Input tensor of shape (B, Nwin, F)

        Returns:
            Logits of shape (B, num_classes)
        """
        z = self.encode(x)       # (B, D)
        logits = self.head(z)    # (B, num_classes)
        return logits


# ======================================
# Mode 3: per-window MLP over features F
# ======================================
class MLPWindow(nn.Module):
    """
    MLP baseline for per-window classification (mode 3).

    Expected input:
        x: (B, F)
           - one feature vector per window

    Output:
        logits: (B, num_classes)

    Notes:
        This is a simple baseline for the flattened-window setting, where each
        window is treated as an independent sample.
    """
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        hidden: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Two hidden layers with GELU and dropout for regularization
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x: Input tensor of shape (B, F)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.net(x)


# ==========================
# Factory for the three modes
# ==========================
def build_model(
    mode: int,
    num_channels: int = 32,
    feature_dim: Optional[int] = None,
    num_classes: int = 2,
):
    """
    Return a model compatible with the selected dataset mode.

    Args:
        mode:
            0/1 -> time-domain EEG
            2   -> per-trial sequence of feature windows
            3   -> per-window features
        num_channels:
            Number of EEG channels for mode 0/1 CNN models.
        feature_dim:
            Feature dimension F.
            Required for modes 2 and 3.
        num_classes:
            Number of output classes.

    Returns:
        A `torch.nn.Module` matching the input format of the selected mode.

    Mode mapping:
        mode 0/1 -> TripathiSmallCNN1D
        mode 2   -> EEGFeatureTransformer
        mode 3   -> MLPWindow
    """
    # Raw / filtered time-domain EEG
    if mode in (0, 1):
        return TripathiSmallCNN1D(num_channels=num_channels, num_classes=num_classes)

    # Sequence of per-window features
    elif mode == 2:
        assert feature_dim is not None, "feature_dim is required for mode=2"
        return EEGFeatureTransformer(feat_dim=feature_dim, num_classes=num_classes)

    # Single-window feature vectors
    elif mode == 3:
        assert feature_dim is not None, "feature_dim is required for mode=3"
        return MLPWindow(feature_dim=feature_dim, num_classes=num_classes)

    else:
        raise ValueError("mode must be one of {0,1,2,3}")