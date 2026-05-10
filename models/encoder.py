# ============================================================
# Vision Transformer Encoder
# ============================================================

import timm
import torch


def build_encoder(device):

    """
    Build Vision Transformer encoder

    Args:
        device: cuda or cpu

    Returns:
        encoder model
    """

    encoder = timm.create_model(

        'vit_small_patch16_224',

        pretrained=False,

        num_classes=0
    )

    encoder = encoder.to(device)

    return encoder
