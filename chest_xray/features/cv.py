import numpy as np
import pandas as pd
import pickle
import torch
from collections.abc import Iterator, Callable
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from os.path import abspath, dirname, exists, normpath, join
from .read_lists import get_data, get_train_val_data, get_test_data, get_bbox_data
from .dataset import XrayDataset
from sklearn.model_selection import KFold
from chest_xray.tools.globals import SEED



#  ▖▖  ▜         ▄▖      ▗ ▘
#  ▙▌█▌▐ ▛▌█▌▛▘  ▙▖▌▌▛▌▛▘▜▘▌▛▌▛▌▛▘
#  ▌▌▙▖▐▖▙▌▙▖▌   ▌ ▙▌▌▌▙▖▐▖▌▙▌▌▌▄▌
#        ▌

def get_image_path() -> str:
    """Get image path relative to this script"""
    script_path: str = dirname(abspath(__file__))
    image_path: str = normpath(join(script_path, "../../data/images/"))
    return image_path


def fetch_data() -> pd.DataFrame:
    """
    Return dataframe with the image name,
    diseases (split by '|'), and patient ID
    """
    return get_data()[["img_name", "diseases", "patient_id"]]


def make_dataframe() -> pd.DataFrame:
    """
    Merge the classification data with the bounding box data into a single dataframe
    """
    bbox_data = get_bbox_data()

    try:
        bbox_pivot = bbox_data.pivot(index='img_name', columns='disease', values=['x', 'y', 'w', 'h'])
    except Exception as e:
        bbox_pivot = bbox_data.pivot_table(index='img_name', columns='disease', values=['x', 'y', 'w', 'h'], aggfunc='first')
    
    bbox_pivot.columns = [f"{metric}_{disease.lower()}" for metric, disease in bbox_pivot.columns]
    bbox_pivot = bbox_pivot.reset_index()
    classification_data = fetch_data()    
    all_data = pd.merge(classification_data, bbox_pivot, on='img_name', how='left')
    # some diseases don't have any bounding box associated with them, to keep everything uniform, we'll add them anyways
    extra_classes = ['hernia', 'fibrosis', 'edema', 'emphysema', 'pleural_thickening', 'consolitation', 'no finding']
    
    bbox_diseases = [d.lower() for d in bbox_data['disease'].unique()]
    all_diseases = sorted(list(set(bbox_diseases + extra_classes)))
    
    bbox_cols = []
    for disease in all_diseases:
        for metric in ['x', 'y', 'w', 'h']:
            bbox_cols.append(f"{metric}_{disease}")
    
    # Create columns for diseases without bounding boxes
    for col in bbox_cols:
        if col not in all_data.columns:
            all_data[col] = 0.0
    
    all_data[bbox_cols] = all_data[bbox_cols].fillna(0.0)
    final_cols = ['img_name', 'diseases', 'patient_id'] + bbox_cols
    all_data = all_data[final_cols]
    return all_data


def split_bbox(data: pd.DataFrame):
    train: pd.DataFrame = data[
        data["img_name"].isin(get_train_val_data()[0])
    ].copy()
    test: pd.DataFrame = data[
        data["img_name"].isin(get_test_data()[0])
    ].copy()
    bbox_patients = data[
        data["img_name"].isin(get_bbox_data()["img_name"])
    ]["patient_id"].unique()
    rng = np.random.default_rng(SEED)
    bbox_patients = rng.permutation(bbox_patients)
    split = int(len(bbox_patients) * 0.6)
    train_bbox_patients = set(bbox_patients[:split])
    bbox_rows = test[
        test["patient_id"].isin(train_bbox_patients)
    ]
    train = pd.concat([train, bbox_rows])
    test = test[~test["patient_id"].isin(train_bbox_patients)]
    # print(len(train) / len(data) * 100)
    return train, test
    



def calculate_stats(loader):
    """
    Calculate mean and standard deviation on images
    to do normalization later
    """
    sum_: float = 0.0
    sum_sq: float = 0.0
    num_pixels: int = 0
    for images, _ in loader:
        sum_ += images.mean(dim = [0, 2, 3]) * images.shape[0]
        sum_sq += (images**2).mean(dim = [0, 2, 3]) * images.shape[0]
        num_pixels += images.shape[0]
    mean: float = sum_ / num_pixels
    std: float = torch.sqrt((sum_sq / num_pixels) - (mean**2))
    return mean, std


def get_normalization_stats(loader):
    """Check if a pickle with mean and std already exists"""
    script_path: str = dirname(abspath(__file__))
    pickle_path: str = normpath(
        join(script_path, "../data/pickles/mean_std.pkl")
    )
    if exists(pickle_path):
        print(f"Loading image mean and std from {pickle_path}")
        with open(pickle_path, "rb") as f:
            stats: dict[str, float] = pickle.load(f)
        return stats['mean'], stats['std']
    print("Calculating image mean and std")
    mean, std = calculate_stats(loader)
    stats: dict[str, float] = {'mean': mean, 'std': std}
    with open(pickle_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved mean and std to {pickle_path}")
    return mean, std


#  ▄   ▗     ▖      ▌
#  ▌▌▀▌▜▘▀▌  ▌ ▛▌▀▌▛▌█▌▛▘
#  ▙▘█▌▐▖█▌  ▙▖▙▌█▌▙▌▙▖▌

class XrayCV:
    """
    Contains both train/val and test data
    """
    def __init__(self, batch_size, seed, num_workers, k_folds) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.k_folds = k_folds
        self.data: pd.DataFrame = make_dataframe()
        self.data["img_path"] = get_image_path() + "/" + self.data["img_name"]
        self.train, self.test = split_bbox(self.data)
        self.diseases: list[str] = sorted(
            self.data["diseases"].str.split("|").explode().unique()
        )



    def fold_loaders(
        self,
        transform = None
    ) -> Iterator[tuple[DataLoader, DataLoader]]:
        """
        Do K-fold cross-validation
        Yield train data loader and val data loader for each fold
        """
        kf: KFold = KFold(n_splits = self.k_folds, shuffle = True, random_state = self.seed)
        unique_patients: np.ndarray = self.train["patient_id"].unique()

        for train_idx, val_idx in kf.split(unique_patients):
            train_ids: np.ndarray = unique_patients[train_idx]
            val_ids: np.ndarray = unique_patients[val_idx]
            t_df: pd.DataFrame = self.train[
                self.train["patient_id"].isin(train_ids)
            ]
            v_df: pd.DataFrame = self.train[
                self.train["patient_id"].isin(val_ids)
            ]
            train: XrayDataset = XrayDataset(
                t_df,
                transform = transform,
                diseases = self.diseases
            )
            val: XrayDataset = XrayDataset(
                v_df,
                transform = transform,
                diseases = self.diseases
            )
            yield (
                DataLoader(
                    train,
                    batch_size = self.batch_size,
                    shuffle = True,
                    num_workers = self.num_workers,
                    pin_memory=torch.cuda.is_available(),
                ),
                DataLoader(
                    val,
                    batch_size = self.batch_size,
                    shuffle = False,
                    num_workers = self.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
            )


    def test_loaders(self, transform = None) -> tuple[XrayDataset, XrayDataset]:
        train_set = XrayDataset(self.train, transform = transform, diseases=self.diseases)
        test_set = XrayDataset(self.test, transform=None, diseases=self.diseases)
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader



if __name__ == "__main__":
    split_bbox()