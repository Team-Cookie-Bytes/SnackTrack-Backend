import os

import torchvision
import torch
from src.datasets.ingredient_dataset import IngredientDataset

DATASET_DIR = "../../data/nutrition5k_dataset_nosides/"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
IMG_DIR = os.path.join(PROCESSED_DIR, "imagery")

INGREDIENTS_PATH = os.path.join(PROCESSED_DIR, "ingredients_metadata.csv")
DISHES_PATH = os.path.join(PROCESSED_DIR, "dishes_info.csv")


def main():
    dataset = IngredientDataset(
        img_dir=IMG_DIR, ingredients_path=INGREDIENTS_PATH, dish_info_path=DISHES_PATH
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = dataset.get_num_of_classes()

    model = torchvision.models.mobilenet_v3_small(
        pretrained=True, num_classes=num_classes
    )


if __name__ == "__main__":
    main()
