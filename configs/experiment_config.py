# ============================================================
# Experiment Configuration
# ============================================================

# Dataset
IMAGE_SIZE = 224

BATCH_SIZE = 32


# Training
LEARNING_RATE = 1e-4

EPOCHS = 3


# Model
PATCH_SIZE = 16

FEATURE_DIM = 384


# Masking
DEFAULT_MASK_RATIO = 0.75


# Experiment Ratios
MASK_RATIOS = [

    0.5,
    0.75,
    0.9
]
