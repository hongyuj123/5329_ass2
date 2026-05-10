# ============================================================
# Evaluate Model Performance
# ============================================================

import torch
from tqdm import tqdm


def evaluate_model(

    encoder,
    classifier,
    testloader,
    device
):

    """
    Evaluate classification accuracy

    Args:
        encoder: trained encoder
        classifier: classification head
        testloader: test dataloader
        device: cuda or cpu

    Returns:
        accuracy
    """

    encoder.eval()

    classifier.eval()

    correct = 0
    total = 0

    progress_bar = tqdm(

        testloader,

        desc="Evaluating"
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device)

            labels = labels.to(device)

            # Extract features
            features = encoder(images)

            # Classification output
            outputs = classifier(features)

            # Prediction
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (

                predicted == labels
            ).sum().item()

            accuracy = 100 * correct / total

            progress_bar.set_postfix({

                "Accuracy": f"{accuracy:.2f}%"
            })

    return accuracy
