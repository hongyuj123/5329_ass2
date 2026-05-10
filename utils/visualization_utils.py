# ============================================================
# Visualization Utility Functions
# ============================================================

import matplotlib.pyplot as plt


def show_original_image(image):

    """
    Visualize original image
    """

    plt.figure(figsize=(4,4))

    plt.imshow(

        image.cpu().permute(1,2,0)
    )

    plt.title("Original Image")

    plt.axis('off')

    plt.show()


def show_masked_image(image):

    """
    Visualize masked image
    """

    plt.figure(figsize=(4,4))

    plt.imshow(

        image.cpu().permute(1,2,0)
    )

    plt.title("Masked Image")

    plt.axis('off')

    plt.show()


def save_image(image, path):

    """
    Save image to figures folder
    """

    plt.figure(figsize=(4,4))

    plt.imshow(

        image.cpu().permute(1,2,0)
    )

    plt.axis('off')

    plt.savefig(

        path,

        bbox_inches='tight'
    )

    plt.close()

    print(f"Image saved to {path}")
