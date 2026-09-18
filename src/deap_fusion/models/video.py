import torch
from torch import nn
from torchvision import models

def _set_bn_eval(module):
    """
    Put BatchNorm layers in evaluation mode.
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


class ResNet18TrialModel(nn.Module):
    """
    ResNet18-based model for trial-level video classification.

    The model:
    - extracts frame-level features using a ResNet18 backbone
    - averages features across frames to get one trial representation
    - classifies the trial with a linear layer

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze all ResNet backbone parameters.
        freeze_bn_stats: If True, keep BatchNorm layers in eval mode during training.
            If None, it follows the value of `freeze_backbone`.
    """
    def __init__(self, num_classes=2, freeze_backbone=True, freeze_bn_stats=None):
        super().__init__()

        # Load pretrained ResNet18 and remove its final classifier.
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)

        self.freeze_backbone = freeze_backbone
        self.freeze_bn_stats = freeze_backbone if freeze_bn_stats is None else freeze_bn_stats

        # Optionally freeze the backbone so only the classifier is trained.
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_bn_eval(self):
        """
        Force all BatchNorm layers in the backbone into evaluation mode.
        """
        self.backbone.apply(_set_bn_eval)

    def train(self, mode=True):
        """
        Override nn.Module.train() to optionally keep BatchNorm layers frozen.

        Args:
            mode: Standard PyTorch train/eval switch.

        Returns:
            self
        """
        super().train(mode)
        if self.freeze_bn_stats:
            self.set_bn_eval()
        return self

    def encode_frames(self, x):
        """
        Encode each frame independently using the ResNet backbone.

        Args:
            x: Input tensor of shape [B, T, C, H, W]

        Returns:
            Frame features of shape [B, T, feat_dim]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1)
        return feats

    def encode_trial(self, x):
        """
        Build one feature vector per trial by averaging frame features.

        Args:
            x: Input tensor of shape [B, T, C, H, W]

        Returns:
            Trial features of shape [B, feat_dim]
        """
        feats = self.encode_frames(x)
        return feats.mean(dim=1)

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x: Input tensor of shape [B, T, C, H, W]

        Returns:
            Logits of shape [B, num_classes]
        """
        trial_feats = self.encode_trial(x)
        return self.classifier(trial_feats)
