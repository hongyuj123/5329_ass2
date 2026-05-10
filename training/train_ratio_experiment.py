# ============================================================
# Train Multiple Mask Ratios
# ============================================================

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder import build_encoder
from masks.random_mask import create_random_mask


# ============================================================
# Device
# ============================================================

device = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(

    root='./data',

    train=True,

    download=True,

    transform=transform
)

trainloader = DataLoader(

    trainset,

    batch_size=32,

    shuffle=True
)


# ============================================================
# Mask Ratios
# ============================================================

mask_ratios = [

    0.5,
    0.75,
    0.9
]


# ============================================================
# Run Experiments
# ============================================================

for ratio in mask_ratios:

    print(f"\nRunning Experiment: Mask Ratio = {ratio}")

    # Build encoder
    encoder = build_encoder(device)

    # Classification head
    classifier = torch.nn.Linear(

        384,
        10
    ).to(device)

    optimizer = torch.optim.Adam(

        list(encoder.parameters()) +
        list(classifier.parameters()),

        lr=1e-4
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Training
    encoder.train()

    classifier.train()

    for epoch in range(2):

        progress_bar = tqdm(

            trainloader,

            desc=f"Ratio {ratio} | Epoch {epoch+1}"
        )

        for images, labels in progress_bar:

            images = images.to(device)

            labels = labels.to(device)

            # Apply masking
            masked_images = create_random_mask(

                images,
                mask_ratio=ratio
            )

            # Extract features
            features = encoder(masked_images)

            # Classification
            outputs = classifier(features)

            # Loss
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            progress_bar.set_postfix({

                "Loss": f"{loss.item():.4f}"
            })

    # Save checkpoint
    torch.save(

        encoder.state_dict(),

        f"checkpoints/encoder_ratio_{ratio}.pth"
    )

    print(f"Experiment {ratio} Finished")
