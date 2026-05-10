# ============================================================
# Train Random Mask Baseline
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
# Build encoder
# ============================================================

encoder = build_encoder(device)


# ============================================================
# Classifier
# ============================================================

classifier = torch.nn.Linear(384, 10).to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(

    list(encoder.parameters()) +
    list(classifier.parameters()),

    lr=1e-4
)

criterion = torch.nn.CrossEntropyLoss()


# ============================================================
# Training Loop
# ============================================================

encoder.train()

classifier.train()

for epoch in range(3):

    correct = 0
    total = 0

    progress_bar = tqdm(

        trainloader,

        desc=f"Epoch {epoch+1}"
    )

    for images, labels in progress_bar:

        images = images.to(device)

        labels = labels.to(device)

        # Apply random masking
        masked_images = create_random_mask(images)

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

        # Accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        progress_bar.set_postfix({

            "Loss": f"{loss.item():.4f}",

            "Accuracy": f"{accuracy:.2f}%"
        })


# ============================================================
# Save checkpoint
# ============================================================

torch.save(

    encoder.state_dict(),

    "checkpoints/encoder_random.pth"
)

print("Training Finished")
