from torchvision import models
import torch
# https://deepwiki.com/andreasveit/densenet-pytorch/6.2-training-configuration
# 161 model has the best top-1 and top-5 accuracy 

NUM_CLASSES=15

device = torch.device("cpu")

densenet = models.densenet161(pretrained = False)
densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, NUM_CLASSES)
densenet = densenet.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)