from ultralytics import YOLO
import torch

import numpy as np
import pandas as pd
import yaml
import cv2

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# 
optimizer_hyper = yaml.safe_load("optimizer.yaml")
# 
loss_hyper = yaml.safe_load("loss.yaml")
# 
amt_hyper = yaml.safe_load("amt.yaml")

model = YOLO("yolov11n.pt")

# 
match optimizer_hyper["optimizer"]:
    case "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_hyper["lr"])
    case "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_hyper["lr"])
    case "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_hyper["lr"])
        
