import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from timm.loss import AsymmetricLossMultiLabel
from tqdm import tqdm

# Using GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# the path the CSV file and to the images
csv_file = "Data_Entry_2017.csv"
image_root = "images"

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

# The transforms of the images for training and validation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449], std=[0.226]),
])

# The class of the dataset, which will be used to load the data and apply the transforms
class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_root, classes, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.classes = classes

        # Mapping the class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.transform = transform

        # finding the path of the images for faster access
        self.image_paths = self.build_image_path_map()

        self.df = self.df[self.df["Image Index"].isin(self.image_paths.keys())].reset_index(drop=True)

    # finding the path of the images for faster access method
    def build_image_path_map(self):
        image_paths = {}

        # Walk through the image root directory and build a mapping of image names to their full paths
        for root, _, files in os.walk(self.image_root):
            for file in files:
                if file.lower().endswith(".png"):
                    image_paths[file] = os.path.join(root, file)

        return image_paths

    # The length of the dataset
    def __len__(self):
        return len(self.df)

    # given an index, this method will return the image and the labels of that image
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get the image name and the label string from the dataframe
        image_name = row["Image Index"]
        label_string = row["Finding Labels"]

        image_path = self.image_paths[image_name]

        image = Image.open(image_path).convert("L")

        labels = torch.zeros(len(self.classes), dtype=torch.float32)

        findings = label_string.split("|")

        # Set the corresponding indices in the labels tensor to 1 for each finding present in the image
        for finding in findings:
            if finding in self.class_to_idx:
                labels[self.class_to_idx[finding]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, labels

# Load CSV and split by patient
df = pd.read_csv(csv_file)

print(f"Total images in CSV: {len(df)}")

# Patient-level split to reduce leakage
unique_patients = df["Patient ID"].unique()

# Setting a random seed for reproducibility and shuffling the unique patient IDs
generator = torch.Generator().manual_seed(seed)
perm = torch.randperm(len(unique_patients), generator=generator)
unique_patients = unique_patients[perm.numpy()]

# Using an 80-20 split for training and validation
train_patient_count = int(0.8 * len(unique_patients))

# creating sets of patient IDs for training and validation
train_patients = set(unique_patients[:train_patient_count])
val_patients = set(unique_patients[train_patient_count:])

# creating training and validation dataframes based on the patient IDs
train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
val_df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)

# Create datasets
train_dataset = ChestXRayDataset(train_df, image_root, classes, transform)
val_dataset = ChestXRayDataset(val_df, image_root, classes, transform)

print(f"Actual train dataset size: {len(train_dataset)}")
print(f"Actual validation dataset size: {len(val_dataset)}")

# DataLoaders
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

# Model
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

# Change VGG16 from 3-channel input to 1-channel input
old_first_layer = model.features[0]

# Create new cnn layer with 1 input channel and keep the rest the same
new_first_layer = nn.Conv2d(
    in_channels=1,
    out_channels=old_first_layer.out_channels,
    kernel_size=old_first_layer.kernel_size,
    stride=old_first_layer.stride,
    padding=old_first_layer.padding,
)

# Initialize the new first layer's weights by averaging the weights of the original 3 channels
with torch.no_grad():
    new_first_layer.weight = nn.Parameter(
        old_first_layer.weight.mean(dim=1, keepdim=True)
    )
    new_first_layer.bias = old_first_layer.bias

model.features[0] = new_first_layer

# Change final classifier output to 15 classes
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes))

# Freeze convolutional feature extractor for head-only training
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

# Loss
criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05)

# Optimizer for head-only training
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=head_lr,
)

# Training functions
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, phase):
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


def validate_one_epoch(model, dataloader, criterion, device, epoch, num_epochs, phase):
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

# Phase 1: Train classifier head
print("\nStarting head-only training...\n")

for epoch in range(head_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, "Head only")
    val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs, "Head only")

    print(f"Epoch [{epoch + 1}/{num_epochs}] "f"Phase: Head only "f"Train Loss: {train_loss:.4f} "f"Val Loss: {val_loss:.4f}")

# Phase 2: Fine-tune full model
print("\nStarting fine-tuning...\n")

for param in model.features.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)

for epoch in range(head_epochs, num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, "Fine-tuning")
    val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs, "Fine-tuning")

    print(f"Epoch [{epoch + 1}/{num_epochs}] "f"Phase: Fine-tuning "f"Train Loss: {train_loss:.4f} "f"Val Loss: {val_loss:.4f}")

# Save model
save_path = "vgg16_chest_xray_asl.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": image_size,
        "num_epochs": num_epochs,
        "head_epochs": head_epochs,
        "head_lr": head_lr,
        "fine_tune_lr": fine_tune_lr,
    },
    save_path,
)
print(f"\nModel saved as {save_path}")