import numpy as np
import pickle
from collections.abc import Iterator
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from os.path import abspath, dirname, exists, normpath, join
from features.read_lists import get_data, get_train_val_data, get_test_data
from features.dataset import XrayDataset
from sklearn.model_selection import KFold

BATCH_SIZE: int = 32    # play around with this on Habrok
SEED: int = 42          # needs a more central location later
NUM_WORKERS: int = 4    # play around with this on Habrok
K_FOLDS: int = 4


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

class XrayLoader:
    def __init__(self) -> None:
        self.data: pd.DataFrame = fetch_data()
        self.data["img_path"] = get_image_path() + "/" + self.data["img_name"]
        train_names: pd.Series = get_train_val_data()[0]
        test_names: pd.Series = get_test_data()[0]
        self.train: pd.DataFrame = self.data[
            self.data["img_name"].isin(train_names)
        ].copy()
        self.test: pd.DataFrame = self.data[
            self.data["img_name"].isin(test_names)
        ].copy()
        self.diseases: list[str] = sorted(
            self.data["diseases"].str.split("|").explode().unique()
        )


    def fold_loaders(
        self,
        k = K_FOLDS, 
        batch_size = BATCH_SIZE,
        transform = None
    ) -> Iterator[tuple[DataLoader, DataLoader]]:
        """
        Do K-fold cross-validation
        Yield train data loader and val data loader for each fold
        """
        kf: KFold = KFold(n_splits = k, shuffle = True, random_state = SEED)
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
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = NUM_WORKERS,
                    pin_memory = True
                ),
                DataLoader(
                    val,
                    batch_size = batch_size,
                    shuffle = False,
                    num_workers = NUM_WORKERS,
                    pin_memory = True
                )
            )

