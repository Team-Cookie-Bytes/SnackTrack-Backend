from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import cv2


class IngredientDataset(Dataset):
    def __init__(self, img_dir: str, ingredients_path: str, dish_info_path: str):
        self.img_dir = img_dir

        self.ingredients_df = pd.read_csv(ingredients_path)
        self.dish_info_df = pd.read_csv(dish_info_path)

    def __len__(self) -> int:
        return len(self.dish_info_df)

    def __getitem__(self, index):
        dish = self.dish_info_df.iloc[index]
        dish_id = dish[0]

        ingredient_ids = self.ingredients_df[self.ingredients_df["dish_id"] == dish_id][
            "ingredient_id"
        ].values
        label_tensor = torch.FloatTensor(ingredient_ids)

        dish_weight = dish[2]
        weight_in_g_tensor = torch.FloatTensor([dish_weight])

        img_path = os.path.join(self.img_dir, dish_id, "rgb.png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = torch.FloatTensor(img)[None, :, :]

        return img_tensor, label_tensor, weight_in_g_tensor

    def get_num_of_classes(self) -> int:
        return self.label_binarizer.classes_.shape[0]
