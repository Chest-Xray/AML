import numpy as np
from os.path import abspath, dirname, normpath, join
import pandas as pd
from read_lists import get_data, get_train_val_data, get_test_data
from torch.utils.data import Dataset
from typing import Callable

#   ▖▖  ▜         ▄▖      ▗ ▘
#   ▙▌█▌▐ ▛▌█▌▛▘  ▙▖▌▌▛▌▛▘▜▘▌▛▌▛▌▛▘
#   ▌▌▙▖▐▖▙▌▙▖▌   ▌ ▙▌▌▌▙▖▐▖▌▙▌▌▌▄▌
#         ▌

def get_image_path():
    """Get image path relative to this script"""
    script_path: str = dirname(abspath(__file__))
    image_path = normpath(join(script_path, "../../data/images/"))
    return image_path


def fetch_data() -> pd.DataFrame:
    """Return dataframe with the image name, diseases (split by '|'), and patient ID"""
    return get_data()[["img_name", "diseases", "patient_id"]]

#   ▖▖          ▄   ▗       ▗
#   ▚▘▄▖▛▘▀▌▌▌  ▌▌▀▌▜▘▀▌▛▘█▌▜▘
#   ▌▌  ▌ █▌▙▌  ▙▘█▌▐▖█▌▄▌▙▖▐▖
#           ▄▌

class XrayDataset(Dataset):
    def __init__(self, data: pd.DataFrame = fetch_data()) -> None:
        self.data = data


    def __len__(self) -> int:
        return len(self.train) + len(self.test)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self.data.iloc[[idx]]


    @property
    def patient_ids(self) -> np.ndarray:
        return self.data["patient_id"].unique()

    @property
    def diseases(self) -> pd.arrays.StringArray:
        return (
            self.data["diseases"]
            .str.split("|")
            .explode()
            .str.strip()
            .unique()
        )

    @property
    def test(self) -> pd.DataFrame:
        test_images = get_test_data()
        test_data = self.data[self.data["img_name"].isin(test_images[0])]
        test_data["img_path"] = get_image_path() + test_data["img_name"]
        return test_data[["patient_id", "img_path", "diseases"]].copy()

    @property
    def train(self) -> pd.DataFrame:
        train_images = get_train_data()
        train_data = self.data[self.data["img_name"].isin(train_images[0])]
        train_data["img_path"] = get_image_path() + train_data["img_name"]
        return train_data[["patient_id", "img_path", "diseases"]].copy()




if __name__ == "__main__":
    d = XrayDataset()
    print(d.test)
