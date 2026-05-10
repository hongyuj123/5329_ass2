# ============================================================
# Random Patch Masking
# ============================================================

import torch


def create_random_mask(images, mask_ratio=0.75):

    """
    Apply random patch masking

    Args:
        images: input images
        mask_ratio: percentage of masked patches

    Returns:
        masked_images
    """

    B, C, H, W = images.shape

    patch_size = 16

    masked_images = images.clone()

    num_patches = H // patch_size

    total_patches = num_patches * num_patches

    num_mask = int(total_patches * mask_ratio)

    for b in range(B):

        mask_indices = torch.randperm(total_patches)[:num_mask]

        for idx in mask_indices:

            row = idx // num_patches
            col = idx % num_patches

            masked_images[
                b,
                :,
                row*patch_size:(row+1)*patch_size,
                col*patch_size:(col+1)*patch_size
            ] = 0

    return masked_images
