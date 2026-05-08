import numpy as np
from os.path import abspath, dirname, normpath, join
import pandas as pd
from read_lists import get_data, get_train_val_data, get_test_data
from torch.utils.data import Dataset
from typing import Callable


def get_image_path():
    """Get image path relative to this script"""
    script_path: str = dirname(abspath(__file__))
    image_path = normpath(join(script_path, "../../data/images/"))
    return image_path


def fetch_data() -> pd.DataFrame:
    """Return dataframe with the image name, diseases (split by '|'), and patient ID"""
    return get_data()[["img_name", "diseases", "patient_id"]]


def fetch_train_data() -> pd.DataFrame:
    """Return dataframe with patient ID, image path, and diseases for training"""
    train_images = get_train_val_data()
    all_data = fetch_data()
    train_data = all_data[all_data["img_name"].isin(train_images[0])]
    train_data["img_path"] = get_image_path() + train_data["img_name"]
    return train_data[["patient_id", "img_path", "diseases"]].copy()


def fetch_test_data() -> pd.DataFrame:
    """Return dataframe with patient ID, image path, and diseases for testing"""
    test_images = get_test_data()
    all_data = fetch_data()
    test_data = all_data[all_data["img_name"].isin(test_images[0])]
    test_data["img_path"] = get_image_path() + test_data["img_name"]
    return test_data[["patient_id", "img_path", "diseases"]].copy()



class XrayDataset(Dataset):
    def __init__(self, data: pd.DataFrame = fetch_data()) -> None:
        self.data = data


    def __len__(self) => int:
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



if __name__ == "__main__":
    d = XrayDataset()
    print(type(d[0]))
