import os
import pandas as pd
from read_lists import get_bbox_data, get_data, get_train_val_data, get_test_data


def get_image_path():
    """Get image path relative to this script"""
    script_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.normpath(os.path.join(script_path, "../../data/images/"))
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


if __name__ == "__main__":
    print(fetch_data())
