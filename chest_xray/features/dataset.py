import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

#   ▖▖          ▄   ▗       ▗
#   ▚▘▄▖▛▘▀▌▌▌  ▌▌▀▌▜▘▀▌▛▘█▌▜▘
#   ▌▌  ▌ █▌▙▌  ▙▘█▌▐▖█▌▄▌▙▖▐▖
#           ▄▌

class XrayDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform = None, 
        diseases = None
    ) -> None:
        self.data: pd.DataFrame = dataframe.reset_index(drop = True)
        self.transform = transform
        if diseases is None:
            self.diseases: list[str] = sorted(
                self.data["diseases"].str.split("|").explode().unique()
            )
        else:
            self.diseases: list[str] = diseases
        self.disease_to_idx: dict[str, int] = {
            disease: i for i, disease in enumerate(self.diseases)
        }


    def __len__(self) -> int:
        """Returns amount of samples in the dataset"""
        return len(self.data)


    def __getitem__(self, idx: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """Get item from dataset with dataset[n]"""
        row: pd.Series = self.data.iloc[idx]
        image: Image.Image = Image.open(row["img_path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        n = len(self.diseases)
        classification_target = torch.zeros(n)
        bbox_target = torch.zeros(n, 4)  # (x, y, w, h) per disease
        for disease in row["diseases"].split("|"):
            if disease not in self.disease_to_idx:
                continue
            i = self.disease_to_idx[disease]
            classification_target[i] = 1.0
            d = disease.lower()
            x = row.get(f"x_{d}", None)
            if pd.notna(x):
                bbox_target[i] = torch.tensor([
                    row[f"x_{d}"],
                    row[f"y_{d}"],
                    row[f"w_{d}"],
                    row[f"h_{d}"],
                ], dtype=torch.float32)
        return image, classification_target, bbox_target
