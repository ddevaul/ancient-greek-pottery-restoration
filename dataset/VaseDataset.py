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
    def __init__(self, root_dir: str, captions_file: str = "captions.csv", transform=None):
        """
        Vase Dataset for loading masked images, full images, masks, and captions.

        Args:
            root_dir (str): Path to the dataset root directory (e.g., /dataset/train or /dataset/val).
            captions_file (str): CSV file containing textual captions for images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.root_dir = root_dir
        self.masked_dir = os.path.join(root_dir, "masked")
        self.full_dir = os.path.join(root_dir, "full")
        self.masks_dir = os.path.join(root_dir, "masks")

        # Load captions
        self.captions = pd.read_csv(os.path.join(root_dir, captions_file))
        self.transform = transform

        # List all masked image file names
        self.masked_images = os.listdir(self.masked_dir)

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
        mask_name = masked_image_name.replace("_masked_", "_mask_").replace(".png", ".pt")
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
            "full_images": full_image,      # RGB image
            "masks": mask,                  # Binary mask tensor
            "text": caption,                # Caption (string)
        }
