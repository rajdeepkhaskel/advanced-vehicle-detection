import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import os

# Define the custom backbone using EfficientNet-B4
class EfficientNetB4Backbone(nn.Module):
    def __init__(self):
        super(EfficientNetB4Backbone, self).__init__()
        efficientnet_b4 = models.efficientnet_b4(pretrained=True)
        self.backbone = nn.Sequential(
            *list(efficientnet_b4.children())[:-2]
        )

    def forward(self, x):
        return self.backbone(x)
    
# Load the model
model = YOLO('jameslahm/yolov10x')

# Configuration dictionary
config = {
    'data': '/path/to/config.yaml',  # Path to the dataset configuration file
    'epochs': 50,                                    
    'batch': 32,                                     
    'augment': True,                                
    'optimizer': 'SGD',                              
    'lr0': 0.01,                                    
    'lrf': 0.001,                                    
    'momentum': 0.9,                                 
    'patience': 50                                   
}

model.model.backbone = EfficientNetB4Backbone()
model.overrides.update(config)

# Train the model
results = model.train(
    data=config['data'],
    epochs=config['epochs'],
    batch=config['batch'],
    augment=config['augment'],
    optimizer=config['optimizer'],
    lr0=config['lr0'],
    lrf=config['lrf'],
    patience=config['patience'],
    pretrained=True
)

# Validate the model
metrics = model.val(data=config['data'], save_json=True)
print("Validation Results:")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"F1 Score: {(2 * metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.75: {metrics.box.map75:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"List of mAP50-95 for each category: {metrics.box.maps}")