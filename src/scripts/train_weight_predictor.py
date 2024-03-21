import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image

from src.datasets.ingredient_dataset import IngredientDataset
from src.models.ResNetRegression import ResNetRegression

DATASET_DIR = "../../data/nutrition5k_dataset_nosides/"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
IMG_DIR = os.path.join(PROCESSED_DIR, "imagery")

INGREDIENTS_PATH = os.path.join(PROCESSED_DIR, "ingredients_metadata.csv")
DISHES_PATH = os.path.join(PROCESSED_DIR, "dishes_info.csv")


def main():
    dataset = IngredientDataset(
        img_dir=IMG_DIR, ingredients_path=INGREDIENTS_PATH, dish_info_path=DISHES_PATH
    )

    model = ResNetRegression()

    # For regression, Mean Squared Error (MSE) is a common loss function
    criterion = nn.MSELoss()

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _, weights in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, weights.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_loader)}")


if __name__ == "__main__":
    main()
