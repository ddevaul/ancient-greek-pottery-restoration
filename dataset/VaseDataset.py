import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd


class VaseDataset(Dataset):
    def __init__(
        self,
        dataset_root_dir: str,
        agg_data_file_name: str = "aggregated_data.csv",
        mask_mappings_file_path: str = "aggregated_data.csv",
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
        self.dataset_root_dir = dataset_root_dir
        self.original_dir = os.path.join(dataset_root_dir, "original_images")
        self.masks_dir = os.path.join(dataset_root_dir, "masks")

        # Load Captions / Metadata dataframe
        self.agg_data_file_path = os.path.join(dataset_root_dir, agg_data_file_name)
        self.aggregate_data_df = pd.read_csv()

        # Load mask -> full image mappings dataframe
        self.mask_mappings_file_path = os.path.join(
            dataset_root_dir, mask_mappings_file_path
        )

        self.mask_mappings_df = pd.read_csv(
            os.path.join(dataset_root_dir, agg_data_file_name)
        )

        self.transform = transform

    def __len__(self):
        return len(self.masked_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get masked image file name
        masked_image_name = self.masked_images[idx]

        # Load masked image
        masked_image_path = os.path.join(self.masked_dir, masked_image_name)
        masked_image = Image.open(masked_image_path).convert("RGB")

        # Derive full image file name from masked image name
        full_image_name = masked_image_name.split("_masked_")[0] + ".jpg"
        full_image_path = os.path.join(self.full_dir, full_image_name)
        # print(full_image_path)
        full_image = Image.open(full_image_path).convert("RGB")

        # Get mask tensor name
        mask_name = masked_image_name.replace("_masked_", "_mask_").replace(
            ".png", ".pt"
        )
        mask_path = os.path.join(self.masks_dir, mask_name)
        # print(mask_path)
        mask = torch.load(mask_path)  # Load the mask tensor (H, W)

        # Find the caption for the full image
        caption_row = self.captions[self.captions["filename"] == full_image_name]
        caption = caption_row["caption"].values[0] if not caption_row.empty else ""

        # Apply transformations if specified
        if self.transform:
            masked_image = self.transform(masked_image)
            full_image = self.transform(full_image)
            # For masks, ensure they are normalized and have an added channel dimension (1, H, W)
            mask = torch.unsqueeze(mask, 0)  # Add channel dimension

        return {
            "masked_images": masked_image,  # RGB image
            "full_images": full_image,  # RGB image
            "masks": mask,  # Binary mask tensor
            "text": caption,  # Caption (string)
        }
