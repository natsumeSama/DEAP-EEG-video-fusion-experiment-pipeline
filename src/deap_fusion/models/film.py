import torch
from torch import nn

# =========================
# FUSION MODEL
# =========================

class FiLMGenerator(nn.Module):
    """
    Generate FiLM parameters from the EEG embedding.

    Role:
        EEG embedding -> gamma, beta

    These parameters are then used to modulate the video representation.
    """
    def __init__(self, eeg_dim, video_dim, hidden_dim=256, dropout=0.2):
        """
        Args:
            eeg_dim: Dimension of EEG embedding
            video_dim: Dimension of video embedding to modulate
            hidden_dim: Hidden size of the FiLM MLP
            dropout: Dropout used inside the FiLM generator
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * video_dim),
        )

    def forward(self, eeg_z):
        """
        Generate FiLM parameters from EEG embedding.

        Args:
            eeg_z: EEG embedding of shape (B, eeg_dim)

        Returns:
            gamma: Multiplicative modulation of shape (B, video_dim)
            beta: Additive modulation of shape (B, video_dim)
        """
        film_params = self.net(eeg_z)
        gamma_raw, beta_raw = torch.chunk(film_params, 2, dim=-1)

        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta = 0.1 * torch.tanh(beta_raw)

        return gamma, beta


class EEGVideoFiLMFusion(nn.Module):
    """
    EEG-video fusion model using FiLM-style conditioning.

    Pipeline:
        EEG   -> eeg_encoder.encode(...)         -> eeg_z
        Video -> video_encoder.encode_trial(...) -> video_z
        eeg_z -> FiLMGenerator                   -> gamma, beta
        video_z modulated by gamma/beta          -> video_mod
        classifier(video_mod)                    -> logits
    """
    def __init__(
        self,
        eeg_encoder: nn.Module,
        video_encoder: nn.Module,
        eeg_dim: int,
        video_dim: int,
        num_classes: int = 2,
        film_hidden_dim: int = 256,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            eeg_encoder: Pretrained EEG encoder
            video_encoder: Pretrained video encoder
            eeg_dim: Dimension of EEG embedding
            video_dim: Dimension of video embedding
            num_classes: Number of output classes
            film_hidden_dim: Hidden size inside FiLM generator
            head_hidden_dim: Hidden size of fusion classifier
            dropout: Dropout used in fusion blocks
        """
        super().__init__()

        self.eeg_encoder = eeg_encoder
        self.video_encoder = video_encoder

        self.eeg_dim = eeg_dim
        self.video_dim = video_dim
        self.num_classes = num_classes

        # FiLM block that predicts modulation parameters from EEG features.
        self.film = FiLMGenerator(
            eeg_dim=eeg_dim,
            video_dim=video_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
        )

        # Extra gate to control how much modulation is applied.
        self.gate = nn.Linear(eeg_dim, video_dim)
        nn.init.constant_(self.gate.bias, -2.0)


        # Final classifier operating on EEG-conditioned video representation
        self.fusion_dim = video_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def freeze_encoders(self):
        """
        Freeze both EEG and video encoders.

        Used in stage 1 so only FiLM and classifier layers are trained.
        """
        for p in self.eeg_encoder.parameters():
            p.requires_grad = False
        for p in self.video_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        """
        Unfreeze both EEG and video encoders.

        Useful when switching to a fine-tuning stage.
        """
        for p in self.eeg_encoder.parameters():
            p.requires_grad = True
        for p in self.video_encoder.parameters():
            p.requires_grad = True

    def forward(self, eeg_x, video_x, return_features=False):
        """
        Forward pass of the fusion model.

        Args:
            eeg_x: EEG input, typically shape (B, Nwin, F)
            video_x: Video input, typically shape (B, T, C, H, W)
            return_features: If True, also return intermediate features

        Returns:
            logits if return_features=False

            or

            logits, feature_dict if return_features=True
        """
        eeg_z = self.eeg_encoder.encode(eeg_x)              # (B, eeg_dim)
        video_z = self.video_encoder.encode_trial(video_x)  # (B, video_dim)

        gamma, beta = self.film(eeg_z)

        # Standard FiLM modulation of video features.
        film_out = gamma * video_z + beta

        # Gated residual interpolation between original and FiLM-modulated video features.
        gate = torch.sigmoid(self.gate(eeg_z))
        video_mod = video_z + gate * (film_out - video_z)
        logits = self.classifier(video_mod)

        if return_features:
            return logits, {
                "eeg_z": eeg_z,
                "video_z": video_z,
                "gamma": gamma,
                "beta": beta,
                "gate": gate,
                "video_mod": video_mod,
                "classifier_input": video_mod,
            }

        return logits
