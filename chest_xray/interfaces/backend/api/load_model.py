from torchvision import models, transforms
import torch
from pathlib import Path

CLASSES = (
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

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "models" / "chest_xray_bbox.pth"



def create_model(num_classes: int):
    model = models.densenet161(weights=None)

    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    old_first_layer = model.features[0]

    new_first_layer = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_first_layer.out_channels,
        kernel_size=old_first_layer.kernel_size,
        stride=old_first_layer.stride,
        padding=old_first_layer.padding,
        bias=old_first_layer.bias is not None,
    )

    model.features[0] = new_first_layer

    return model


def get_model(model_path: str = MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(len(CLASSES))

    loaded = torch.load(str(model_path), map_location=device, weights_only=False)

    if isinstance(loaded, dict):
        if "state_dict" in loaded:
            state = loaded["state_dict"]
        else:
            state = loaded
        model.load_state_dict(state)
    else:
        model = loaded

    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    return model, transform, device
