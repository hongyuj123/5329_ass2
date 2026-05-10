# ============================================================
# Save Utility Functions
# ============================================================

import os
import torch


def create_directories():

    """
    Create required project directories
    """

    os.makedirs("checkpoints", exist_ok=True)

    os.makedirs("features", exist_ok=True)

    os.makedirs("figures", exist_ok=True)

    os.makedirs("logs", exist_ok=True)


def save_checkpoint(

    model,
    path
):

    """
    Save model checkpoint
    """

    torch.save(

        model.state_dict(),

        path
    )

    print(f"Checkpoint saved to {path}")


def save_feature(

    feature,
    path
):

    """
    Save extracted features
    """

    torch.save(

        feature.cpu(),

        path
    )

    print(f"Feature saved to {path}")
