from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import os
import numpy as np
import pandas as pd
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
            min(size[0] - margin, cluster_center_x + cluster_radius),
        )
        y1 = np.random.randint(
            max(margin, cluster_center_y - cluster_radius),
            min(size[1] - margin, cluster_center_y + cluster_radius),
        )
        width = np.random.randint(20, (size[0] - margin) // 3)  # Small random width
        height = np.random.randint(20, (size[1] - margin) // 3)  # Small random height
        x2 = min(
            size[0] - margin, x1 + width
        )  # Ensure it doesn't go beyond the right edge
        y2 = min(
            size[1] - margin, y1 + height
        )  # Ensure it doesn't go beyond the bottom edge

        draw.rectangle([x1, y1, x2, y2], fill=255)

    # Convert to a binary tensor (1 for masked, 0 for unmasked)
    mask_tensor = torch.tensor(np.array(mask) > 0, dtype=torch.float32)  # Shape: (H, W)
    return mask_tensor


@dataclass
class BatchMaskArgs:
    original_images_dir: str
    mask_dir: str
    image_names_batch: list[str]
    num_masks_per_image: int
    image_size: list[int, int]


def mask_batch_of_images(args: BatchMaskArgs):
    # Unpack the args
    original_images_dir = args.original_images_dir
    mask_dir = args.mask_dir
    image_names_batch = args.image_names_batch
    num_masks_per_image = args.num_masks_per_image
    image_size = args.image_size

    batch_mappings = []

    for image_name in image_names_batch:
        original_image_path = os.path.join(original_images_dir, image_name)
        orignal_image = Image.open(original_image_path).convert("RGB")
        orignal_image = orignal_image.resize(image_size)

        for i in range(num_masks_per_image):
            # Generate random mask for this image
            mask_tensor = generate_random_mask(image_size)

            # Create and save mask as a black and white png
            mask_image = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8))

            # Save the mask image with naming convention: {original_image_name}_mask_{i}.png
            mask_image_name = f"{image_name.split('.')[0]}_mask_{i}.png"
            mask_image_path = os.path.join(mask_dir, mask_image_name)
            mask_image.save(mask_image_path, format="PNG", optimize=True)

            # Record the mapping
            batch_mappings.append(
                {"original_image_name": image_name, "mask_png": mask_image_name}
            )

    return batch_mappings


def process_images_parallel(
    root_dir, num_masks_per_image=10, image_size=(512, 512), batch_size=200
):
    original_images_dir = os.path.join(root_dir, "original_images")
    mask_dir = os.path.join(root_dir, "masks")
    output_file = os.path.join(root_dir, "mask_mappings.csv")

    # Create directory if it doesn't exits
    os.makedirs(mask_dir, exist_ok=True)

    # Create our batches of image names to pass to the processes
    image_names = os.listdir(original_images_dir)
    image_names_batches = [
        image_names[i : i + batch_size] for i in range(0, len(image_names), batch_size)
    ]

    # Create list of args for each batch
    batch_mask_args_list = [
        BatchMaskArgs(
            original_images_dir=original_images_dir,
            mask_dir=mask_dir,
            image_names_batch=batch,
            num_masks_per_image=num_masks_per_image,
            image_size=image_size,
        )
        for batch in image_names_batches
    ]

    # Use all available CPUs except 2 so our computer isn't bricked
    num_cpus = cpu_count() - 2 if cpu_count() > 4 else 1
    print(f"Generating masks for {len(image_names)} images using {num_cpus} CPUs...")

    # Pass the arguments to the process_single_image function using a cpu pool
    with Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(mask_batch_of_images, batch_mask_args_list),
                total=len(batch_mask_args_list),
                desc="Processing Images",
            )
        )

    # Flatten results and save mappings
    mappings = [mapping for batch_mapping in results for mapping in batch_mapping]
    pd.DataFrame(mappings).to_csv(output_file, index=False)
    print(f"Saved mappings to {output_file}")


if __name__ == "__main__":

    NUM_MASKS_PER_IMAGE = 10  # Number of masks to generate per image
    IMAGE_SIZE = (512, 512)  # Resize all images to this size

    if os.path.exists(
        "./ancient-greek-pottery-restoration/dataset/train/original_images"
    ):
        DATASET_ROOT_DIR = "./ancient-greek-pottery-restoration/dataset/train"
    elif os.path.exists("./dataset/train/original_images"):
        DATASET_ROOT_DIR = "./dataset/train"

    process_images_parallel(
        DATASET_ROOT_DIR,
        num_masks_per_image=NUM_MASKS_PER_IMAGE,
        image_size=IMAGE_SIZE,
        batch_size=200,
    )
