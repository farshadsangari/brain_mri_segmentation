import re
import random
import pandas as pd
from .prepare_data import get_file_list
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image


def dataloader(train_data, test_data, val_data, batch_size):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return [train_loader, val_loader, test_loader]
