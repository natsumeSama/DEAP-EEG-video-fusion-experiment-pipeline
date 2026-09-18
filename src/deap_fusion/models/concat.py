import torch
from torch import nn

# =========================================================
# FEATURE-LEVEL CONCATENATION FUSION BASELINE
# =========================================================

class EEGVideoConcatFusion(nn.Module):
    """
    Feature-level concatenation fusion baseline.

    This model uses the same pretrained EEG and video encoders as FiLM fusion.

    Difference with FiLM:
    - FiLM: EEG conditions/modulates video features.
    - Concat: EEG and video features are simply concatenated.

    This gives a fair static feature-level fusion baseline.
    """

    def __init__(
        self,
        eeg_encoder,
        video_encoder,
        eeg_dim,
        video_dim,
        num_classes=2,
        head_hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.eeg_encoder = eeg_encoder
        self.video_encoder = video_encoder

        self.eeg_dim = eeg_dim
        self.video_dim = video_dim
        self.fusion_dim = eeg_dim + video_dim

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
        Freeze EEG and video encoders.

        Stage 1 trains only the concat classifier.
        """
        for p in self.eeg_encoder.parameters():
            p.requires_grad = False

        for p in self.video_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        """
        Unfreeze EEG and video encoders.
        """
        for p in self.eeg_encoder.parameters():
            p.requires_grad = True

        for p in self.video_encoder.parameters():
            p.requires_grad = True

    def forward(self, eeg_x, video_x, return_features=False):
        """
        Args:
            eeg_x: EEG input, usually (B, Nwin, F)
            video_x: video input, usually (B, T, C, H, W)
            return_features: if True, return intermediate embeddings

        Returns:
            logits or (logits, feature_dict)
        """
        eeg_z = self.eeg_encoder.encode(eeg_x)
        video_z = self.video_encoder.encode_trial(video_x)

        fusion_z = torch.cat([eeg_z, video_z], dim=1)

        logits = self.classifier(fusion_z)

        if return_features:
            return logits, {
                "eeg_z": eeg_z,
                "video_z": video_z,
                "fusion_z": fusion_z,
            }

        return logits