import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


class IngredientDataset(Dataset):
    def __init__(
        self, img_dir: str, ingredients_path: str, dish_info_path: str, transform=None
    ):
        self.img_dir = img_dir

        self.ing_df = pd.read_csv(ingredients_path)
        self.dish_info_df = pd.read_csv(dish_info_path)

        self.transform = transform

        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit([self.ing_df["ingredient_id"].to_list()])

    def __len__(self) -> int:
        return len(self.dish_info_df)

    def __getitem__(self, index):
        dish = self.dish_info_df.iloc[index]
        dish_id = dish[0]
        # print(dish_id)

        ingredient_ids = self.ing_df[self.ing_df["dish_id"] == dish_id][
            "ingredient_id"
        ].values
        label_encoded = self.label_binarizer.transform([ingredient_ids])[0]
        label_tensor = torch.FloatTensor(label_encoded)
        # print(ingredient_ids)

        dish_weight = dish[2]
        # print(dish_weight)
        weight_in_g_tensor = torch.FloatTensor([dish_weight])

        img_path = os.path.join(self.img_dir, dish_id, "rgb.png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.show()

        img_tensor = torch.FloatTensor(img)
        img_tensor = (
            self.transform(img_tensor) if self.transform is not None else img_tensor
        )

        return img_tensor, label_tensor, weight_in_g_tensor

    def get_num_of_classes(self) -> int:
        return self.label_binarizer.classes_.shape[0]
