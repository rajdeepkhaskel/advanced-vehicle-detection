# Advanced-Vehicle-Detection

This repository contains the code and data for the Advanced Vehicle Detection (AVD) Hackathon. The goal of this project is to develop an automatic vehicle detection system that can accurately identify and classify vehicles in images captured under various weather conditions. This system is crucial for Intelligent Transportation Systems (ITS) and smart city applications, including self-driving cars and driver assistance systems. The dataset is available here: [github.com/Sourajit-Maity/juvdv2-vdvwc](https://github.com/Sourajit-Maity/juvdv2-vdvwc).
<br>
For using the dataset, create a folder named `data` in parent directory. Arrange the data by creating 2 subfolders in the `data` folder, named `images` and `labels`. Keep the images in the `images` folder, and the annotations in the `labels` folder.

## Repository Structure

- `best_weights.pt`: **Contains the weights of the BEST model**.
- `best_output.txt`: Contains the text of the best output.
- `other_models/`: **This directory has the model weights and outputs of all other experiments**.
- `config.yaml`: Configuration file for setting up the dataset paths and other parameters.
- `data/`: Folder containing the dataset arranged in YOLO format.
- `main.py`: The main script to run the vehicle detection system.
- `runs/`: Directory for storing **BEST** model outputs and logs.
- `supplementary.py`: Additional script used for supporting tasks.

## Getting Started

### Prerequisites

Ensure you have the following packages installed:

```bash
pip install torch torchvision timm ultralytics
