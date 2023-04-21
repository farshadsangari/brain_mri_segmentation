import os
import re
import numpy as np
import pandas as pd
import re
import albumentations
from torch.utils.data import Dataset
import cv2
import torchvision as tv


def get_file_list(data_path):

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".tif")):
                    data_list.append(os.path.join(subdir, file))
    data_list.sort()
    if not data_list:
        raise FileNotFoundError("No data was found")
    return data_list


class myDataset(Dataset):
    def __init__(self, x, y, transforms=False):
        self.x = x
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = cv2.imread(self.x[index])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y = cv2.imread(self.y[index])
        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

        if self.transforms:
            augmentations = self.transforms
            augmentations = augmentations(image=x, mask=y)
            x_aug = augmentations["image"]
            y_aug = augmentations["mask"]
        else:
            x_aug = x
            y_aug = y
        to_tensor = tv.transforms.Compose([tv.transforms.ToTensor()])

        return to_tensor(x), to_tensor(y), to_tensor(x_aug), to_tensor(y_aug)


def create_dataset(data_paths, regex_image_paths, transforms):

    x_path = get_file_list(data_paths)
    x_path = [data.replace("\\", "/") for data in x_path]
    x_path = [path for path in x_path if re.findall("Output", path) == []]
    r = re.compile(".*_\d*.tif$")
    x_data = list(filter(r.match, x_path))

    df = pd.DataFrame(x_data).reset_index(level=0, drop=True)
    df.columns = ["x_path"]

    df["y_path"] = df["x_path"].apply(lambda x: x[:-4] + "_mask" + x[-4:])
    df["patient_id"] = df["x_path"].apply(
        lambda x: re.search(regex_image_paths, x).group(1)
    )
    df["patient_trial"] = df["x_path"].apply(
        lambda x: re.search(regex_image_paths, x).group(2)
    )

    lst_of_patients = list(df["patient_id"].unique())
    np.random.seed(4)
    np.random.shuffle(lst_of_patients)

    patients_test = lst_of_patients[:11]
    patients_val = lst_of_patients[11:22]
    patients_train = lst_of_patients[22:]

    df_train_path = df[df["patient_id"].isin(patients_train)]
    df_test_path = df[df["patient_id"].isin(patients_test)]
    df_val_path = df[df["patient_id"].isin(patients_val)]

    x_train_directory = list(df_train_path["x_path"].values)
    y_train_directory = list(df_train_path["y_path"].values)

    x_test_directory = list(df_test_path["x_path"].values)
    y_test_directory = list(df_test_path["y_path"].values)

    x_val_directory = list(df_val_path["x_path"].values)
    y_val_directory = list(df_val_path["y_path"].values)

    dataset_train = myDataset(
        x_train_directory, y_train_directory, transforms=transforms
    )
    dataset_val = myDataset(x_val_directory, y_val_directory, transforms=None)
    dataset_test = myDataset(x_test_directory, y_test_directory, transforms=None)
    return dataset_train, dataset_val, dataset_test
