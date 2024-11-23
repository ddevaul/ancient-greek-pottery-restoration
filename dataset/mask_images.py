import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm


def generate_random_mask(size, margin=40, cluster_radius=50):
    """
    Generate a random binary mask with smaller masked regions grouped closer together.
    
    Args:
        size (tuple): The size of the image (width, height).
        margin (int): Minimum distance from the edges for the masks.
        cluster_radius (int): Maximum distance from the cluster center for rectangles.
        
    Returns:
        torch.Tensor: Binary tensor mask (1 for masked regions, 0 for unmasked).
    """
    mask = Image.new("L", size, 0)  # Black background for the mask
    draw = ImageDraw.Draw(mask)

    # Randomly pick a center for the cluster
    cluster_center_x = np.random.randint(margin, size[0] - margin)
    cluster_center_y = np.random.randint(margin, size[1] - margin)

    # Add random rectangles around the cluster center
    for _ in range(np.random.randint(3, 6)):  # Random number of rectangles
        # Generate rectangle coordinates near the cluster center
        x1 = np.random.randint(
            max(margin, cluster_center_x - cluster_radius),
            min(size[0] - margin, cluster_center_x + cluster_radius)
        )
        y1 = np.random.randint(
            max(margin, cluster_center_y - cluster_radius),
            min(size[1] - margin, cluster_center_y + cluster_radius)
        )
        width = np.random.randint(20, (size[0] - margin) // 3)  # Small random width
        height = np.random.randint(20, (size[1] - margin) // 3)  # Small random height
        x2 = min(size[0] - margin, x1 + width)  # Ensure it doesn't go beyond the right edge
        y2 = min(size[1] - margin, y1 + height)  # Ensure it doesn't go beyond the bottom edge

        draw.rectangle([x1, y1, x2, y2], fill=255)

    # Convert to a binary tensor (1 for masked, 0 for unmasked)
    mask_tensor = torch.tensor(np.array(mask) > 0, dtype=torch.float32)  # Shape: (H, W)
    return mask_tensor



def process_images(root_dir, num_masks_per_image=10, image_size=(512, 512)):
    """Process all images in a specified root directory."""
    full_dir = os.path.join(root_dir, "full")
    masked_dir = os.path.join(root_dir, "masked")
    mask_dir = os.path.join(root_dir, "masks")
    output_file = os.path.join(root_dir, "mappings.csv")

    # Create directories if they don't exist
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    mappings = []

    # Iterate through all images in the full directory
    for image_name in tqdm(os.listdir(full_dir), desc=f"Processing images in {root_dir}"):
        full_image_path = os.path.join(full_dir, image_name)

        # Open and resize the full image
        full_image = Image.open(full_image_path).convert("RGB")
        full_image = full_image.resize(image_size)

        for i in range(num_masks_per_image):
            # Generate a random mask tensor
            mask_tensor = generate_random_mask(image_size)

            # Create a masked image with a green background
            green_background = Image.new("RGB", image_size, (0, 255, 0))  # Green background
            mask_image = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8))  # Convert tensor to image
            masked_image = Image.composite(green_background, full_image, mask_image)

            # Save the masked image
            masked_image_name = f"{image_name.split('.')[0]}_masked_{i}.png"
            masked_image_path = os.path.join(masked_dir, masked_image_name)
            masked_image.save(masked_image_path)

            # Save the mask tensor
            mask_tensor_name = f"{image_name.split('.')[0]}_mask_{i}.pt"
            mask_tensor_path = os.path.join(mask_dir, mask_tensor_name)
            torch.save(mask_tensor, mask_tensor_path)

            # Record the mapping
            mappings.append({
                "masked_image": masked_image_name,
                "full_image": image_name,
                "mask_tensor": mask_tensor_name
            })

    # Save mappings to CSV
    pd.DataFrame(mappings).to_csv(output_file, index=False)
    print(f"Saved mappings to {output_file}")


if __name__ == "__main__":
    # Parameters
    NUM_MASKS_PER_IMAGE = 10  # Number of masks to generate per image
    IMAGE_SIZE = (512, 512)  # Resize all images to this size
    ROOT_DIRS = ["dataset/train", "dataset/val"]  # Directories to process

    # Process each root directory
    for root_dir in ROOT_DIRS:
        process_images(root_dir, num_masks_per_image=NUM_MASKS_PER_IMAGE, image_size=IMAGE_SIZE)
