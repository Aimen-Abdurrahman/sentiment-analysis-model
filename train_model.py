import torch
import torch.nn as nn
import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(device)


data = pd.read_csv("EcoPreprocessed.csv", usecols=[1,2], header=None)

print(data.head())