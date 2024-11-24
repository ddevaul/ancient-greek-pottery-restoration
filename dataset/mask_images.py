import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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


def process_single_image(args):

    original_images_dir, masked_dir, mask_dir, image_name, num_masks_per_image, image_size = args
    original_image_path = os.path.join(original_images_dir, image_name)
    orignal_image = Image.open(original_image_path).convert("RGB")
    orignal_image = orignal_image.resize(image_size)

    mappings = []

    for i in range(num_masks_per_image):
        # Generate random mask for this image
        mask_tensor = generate_random_mask(image_size)
        
        # Apply the mask to the image with the color green
        # green_background = Image.new("RGB", image_size, (0, 255, 0))
        mask_image = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8))
        # masked_image = Image.composite(green_background, orignal_image, mask_image)

        # # Save the masked image in format: {original_image_name}_masked_{i}.png
        # masked_image_name = f"{image_name.split('.')[0]}_masked_{i}.png"
        # masked_image_path = os.path.join(masked_dir, masked_image_name)
        # masked_image.save(masked_image_path, format="PNG", optimize=True)

       # Save the mask as a black-and-white PNG
        mask_image_name = f"{image_name.split('.')[0]}_mask_{i}.png"
        mask_image_path = os.path.join(mask_dir, mask_image_name)
        mask_image.save(mask_image_path, format="PNG", optimize=True)

        # Record the mapping
        mappings.append({
            # "masked_image_name": masked_image_name,
            "original_image_name": image_name,
            "mask_png": mask_image_name
        })

    return mappings


def process_images_parallel(root_dir, num_masks_per_image=10, image_size=(512, 512)):
    original_images_dir = os.path.join(root_dir, "original_images")
    masked_dir = os.path.join(root_dir, "masked_images")
    mask_dir = os.path.join(root_dir, "masks")
    
    output_file = os.path.join(root_dir, "mask_mappings.csv")

    # Create directories if they don't exist
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_names = os.listdir(original_images_dir)
    # Use all available CPUs except 2 so we're not bricked
    num_cpus = cpu_count() - 2 if cpu_count() > 4 else 1 
    
    print(f"Generating masks for {len(image_names)} images using {num_cpus} CPUs...")

    # Create an argument tuple for each image
    args_list = [
        (original_images_dir, masked_dir, mask_dir, image_name, num_masks_per_image, image_size)
        for image_name in image_names
    ]

    # Pass the arguments to the process_single_image function using a cpu pool
    with Pool(processes=num_cpus) as pool:
        results = list(tqdm(pool.imap(process_single_image, args_list), total=len(args_list), desc="Processing Images"))

    # Flatten results and save mappings
    mappings = [mapping for partial_mapping in results for mapping in partial_mapping]
    pd.DataFrame(mappings).to_csv(output_file, index=False)
    print(f"Saved mappings to {output_file}")


if __name__ == "__main__":
    # Parameters
    NUM_MASKS_PER_IMAGE = 10  # Number of masks to generate per image
    IMAGE_SIZE = (512, 512)  # Resize all images to this size
    
    if os.path.exists("./ancient-greek-pottery-restoration/dataset/train/original_images"):
        DATASET_ROOT_DIR = "./ancient-greek-pottery-restoration/dataset/train"
    elif os.path.exists("./dataset/train/original_images"):
        DATASET_ROOT_DIR = "./dataset/train"
    
    process_images_parallel(DATASET_ROOT_DIR, num_masks_per_image=NUM_MASKS_PER_IMAGE, image_size=IMAGE_SIZE)
        
        
