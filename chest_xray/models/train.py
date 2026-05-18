import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from chest_xray.data.chestdataset import ChestXRayDataset

# the path the CSV file and to the images


# The 15 output labels of the model
classes = (
    "Hernia",
    "Pneumonia",
    "Fibrosis",
    "Effusion",
    "Edema",
    "Emphysema",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Cardiomegaly",
    "Infiltration",
    "Pleural_Thickening",
    "Consolidation",
    "Pneumothorax",
    "No Finding",
)

# Hyperparameters
batch_size = 16
num_epochs = 30
head_epochs = 5
head_lr = 1e-3
fine_tune_lr = 1e-4
image_size = 512
seed = 42


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def load_csv(self, csv_file):
        Root = Path(__file__).parent.parent
        csv_file = Root / "data" / "lists" / "Data_Entry_2017.csv"
        image_root = Root / "data" / "images"
        return pd.read_csv(csv_file)
    
    def transform_images(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ])
        
    def create_dataloaders(self, df, image_root, classes, transform):
        unique_patients = df["Patient ID"].unique()
        
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(unique_patients), generator=generator)
        
        unique_patients = unique_patients[perm.numpy()]
        train_patient_count = int(0.8 * len(unique_patients))
        
        train_patients = set(unique_patients[:train_patient_count])
        val_patients = set(unique_patients[train_patient_count:])
        
        train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
        val_df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
        
        train_dataset = ChestXRayDataset(train_df, image_root, classes, transform)
        val_dataset = ChestXRayDataset(val_df, image_root, classes, transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader

    def train_one_epoch(self, model, dataloader, criterion, optimizer, device, epoch, num_epochs, phase):
        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Train",
            unit="batch",
            dynamic_ncols=True
        )

        for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / batch_idx

            progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        return running_loss / len(dataloader)


    def validate_one_epoch(self, model, dataloader, criterion, device, epoch, num_epochs, phase):
        model.eval()

        running_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}] {phase} Val",
            unit="batch",
            dynamic_ncols=True
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(progress_bar, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                average_loss = running_loss / batch_idx

                progress_bar.set_postfix(loss=f"{average_loss:.4f}")

        return running_loss / len(dataloader)
