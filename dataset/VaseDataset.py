import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class VaseDataset(Dataset):
    def __init__(
        self,
        dataset_root_dir: str,
        agg_data_file_name: str = "aggregated_data.csv",
        mask_mappings_file_path: str = "mask_mappings.csv",
        transform=None,
    ):
        """
        Vase Dataset for loading masked images, original images, masks, and captions.

        Args:
            dataset_root_dir (str): Path to the root directory of the dataset.
                                    This directory should contain the following:
                                        1. full: Directory containing the original images
            agg_data_file_name (str): Name of the file containing the aggregated data (Should exist in the dataset_root_dir directory!)
            transform: Transformations to apply to images.
        """
        if transform is None:
            raise ValueError("Transforms must be specified!")
        
        self.dataset_root_dir = dataset_root_dir
        self.original_dir = os.path.join(dataset_root_dir, "original_images")
        self.masks_dir = os.path.join(dataset_root_dir, "masks")

        # Load masked images
        self.masked_images_list = os.listdir(self.masks_dir)

        # Load Captions / Metadata dataframe
        self.agg_data_file_path = os.path.join(dataset_root_dir, agg_data_file_name)
        self.aggregate_data_df: pd.DataFrame = pd.read_csv(self.agg_data_file_path)

        # Load mask -> full image mappings dataframe
        self.mask_mappings_file_path = os.path.join(
            dataset_root_dir, mask_mappings_file_path
        )
        self.mask_mappings_df: pd.DataFrame = pd.read_csv(self.mask_mappings_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.masked_images_list)

    def __repr__(self):
        return f"VaseDataset: {len(self)} masked images"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the masked image we want to load in
        masked_image_name = self.masked_images_list[idx]

        # Load the black and white mask
        mask_path = os.path.join(self.masks_dir, masked_image_name)
        mask_png = Image.open(mask_path).convert("L")
        mask_array = (np.array(mask_png) > 0).astype(np.float32)

        # Load in the original image
        original_image_name = (
            self.mask_mappings_df.query("mask_png == @masked_image_name")[
                ["original_image_name"]
            ]
            .iloc[0]
            .values
        )[0]

        origal_image_path = os.path.join(self.original_dir, original_image_name)
        original_image = Image.open(origal_image_path).convert("RGB")

        # Apply the mask to the original image (Make the masked region green)
        original_image_array = np.array(original_image)
        masked_image_array = original_image_array.copy()
        print(mask_array.shape)
        print(mask_array)
        masked_image_array[mask_array == 1] = [
            0,
            255,
            0,
        ]  # wherever the mask is 1 make this pixel green

        # convert the masked image back to PIL format
        masked_image = Image.fromarray(masked_image_array)

        # Get the caption information
        pottery_style, pottery_shape = (
            self.aggregate_data_df.query("uniform_image_name == @original_image_name")[
                ["pottery_style", "pottery_shape"]
            ]
            .iloc[0]
            .values
        )

        caption = (
            f"Ancient Greek Pottery. Style: {pottery_style}, Shape: {pottery_shape}."
        )

        # Apply transformations to get PyTorch tensors
        original_image = self.transform(original_image)
        masked_image = self.transform(masked_image)
        mask_tensor = self.transform(mask_png)
        
        return {
            "masked_images": masked_image,  # RGB image with green mask applied
            "full_images": original_image,  # Original RGB image
            "masks": mask_tensor,           # Binary mask tensor (1 for masked regions)
            "text": caption,                # Caption (string)
        }
