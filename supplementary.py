# Other backbones used, and with their code

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import os
    
# Load the model
model = YOLO('jameslahm/yolov10x') # model = YOLO('jameslahm/yolov10{n/s/m/b/l/x}')

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
    
model.model.backbone = EfficientNetB4Backbone()


# Other backbones and their integration in the code

# class ResNet50Backbone(nn.Module):
#     def __init__(self):
#         super(ResNet50Backbone, self).__init__()
#         resnet50 = models.resnet50(pretrained=True)
#         self.backbone = nn.Sequential(
#             *list(resnet50.children())[:-2]
#         )
#     def forward(self, x):
#         return self.backbone(x)
# model.model.backbone = ResNet50Backbone()


# class ResNet152Backbone(nn.Module):
#     def __init__(self):
#         super(ResNet152Backbone, self).__init__()
#         resnet152 = models.resnet152(pretrained=True)
#         self.backbone = nn.Sequential(
#             *list(resnet152.children())[:-2]
#         )
#     def forward(self, x):
#         return self.backbone(x)
# model.model.backbone = ResNet152Backbone()


# class MobileNetV2Backbone(nn.Module):
#     def __init__(self):
#         super(MobileNetV2Backbone, self).__init__()
#         mobilenet_v2 = models.mobilenet_v2(pretrained=True)
#         self.backbone = nn.Sequential(
#             *list(mobilenet_v2.features)
#         )
#     def forward(self, x):
#         return self.backbone(x)
# model.model.backbone = MobileNetV2Backbone()


# class EfficientNetB5Backbone(nn.Module):
#     def __init__(self):
#         super(EfficientNetB5Backbone, self).__init__()
#         efficientnet_b5 = models.efficientnet_b5(pretrained=True)
#         self.backbone = nn.Sequential(
#             *list(efficientnet_b5.children())[:-2]
#         )
#     def forward(self, x):
#         return self.backbone(x)
# model.model.backbone = EfficientNetB5Backbone()

# import timm
# class SwinTransformerBackbone(nn.Module):
#     def __init__(self):
#         super(SwinTransformerBackbone, self).__init__()
#         self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
#     def forward(self, x):
#         x = self.backbone(x)
#         return x[-1]
# model.model.backbone = SwinTransformerBackbone()


# Other heads and their integration in code

# class ResNet50Head(nn.Module):
#     def __init__(self):
#         super(ResNet50Head, self).__init__()
#         resnet50 = models.resnet50(pretrained=True)
#         self.head = nn.Sequential(
#             *list(resnet50.children())[-3:]
#         )
#     def forward(self, x):
#         return self.head(x)
# model.model.head = ResNet50Head()


# class EfficientNetB4Head(nn.Module):
#     def __init__(self):
#         super(EfficientNetB4Head, self).__init__()
#         efficientnet_b4 = models.efficientnet_b4(pretrained=True)
#         self.head = nn.Sequential(
#             *list(efficientnet_b4.children())[-3:]
#         )
#     def forward(self, x):
#         return self.head(x)
# model.model.head = EfficientNetB4Head()


# class MobileNetV2Head(nn.Module):
#     def __init__(self):
#         super(MobileNetV2Head, self).__init__()
#         mobilenet_v2 = models.mobilenet_v2(pretrained=True)
#         self.head = nn.Sequential(
#             *list(mobilenet_v2.children())[-3:]
#         )
#     def forward(self, x):
#         return self.head(x)
# model.model.head = MobileNetV2Head()


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