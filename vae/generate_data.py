import os
import sys
from typing import NamedTuple, Tuple
import scipy
import wget

import numpy as np
import torch as T
from h5py import File
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


# Set the random seed
proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, proj_path)

from config.config import config as c


class VaeDatasets(NamedTuple):
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    training: Dataset


def load_mat_data(file_path):
    print(f"Loading data from {file_path}")
    data = scipy.io.loadmat(file_path)
    images = data["X"]  # Shape (32, 32, 3, N)
    labels = data["y"]  # Shape (N, 1)

    # NaN Handling (Imputation with mean)
    mean_image = np.nanmean(images, axis=(0, 1, 2, 3), keepdims=True)
    images = np.nan_to_num(images, nan=mean_image)

    # Normalization (after NaN handling!)
    images = np.transpose(images, (3, 2, 0, 1)) / 255.0

    labels = labels.flatten()
    labels[labels == 10] = 0

    print("Input image shape:", images.shape)
    print("Input label shape:", labels.shape)

    return images, labels  # Return images and labels as NumPy arrays


def get_house_data() -> VaeDatasets:
    get_data_if_not_exist()
    batch_size = c.get_batch_size()
    train_file_path = data_path + c.get_train_info()[1]
    test_file_path = data_path + c.get_test_info()[1]

    # Load and clean data
    images_train, labels_train = load_mat_data(train_file_path)  # Get NumPy arrays
    images_test, labels_test = load_mat_data(test_file_path)  # Get NumPy arrays

    # Convert to tensors AFTER cleaning and loading
    images_train = T.tensor(images_train, dtype=T.float32)
    labels_train = T.tensor(labels_train, dtype=T.long)
    images_test = T.tensor(images_test, dtype=T.float32)
    labels_test = T.tensor(labels_test, dtype=T.long)

    train_ds = TensorDataset(images_train, labels_train)
    test_ds = TensorDataset(images_test, labels_test)

    train_len = int(len(train_ds) * 0.7)
    val_len = len(train_ds) - train_len
    lengths = [train_len, val_len]
    trnSet, valSet = random_split(train_ds, lengths)

    train_loader = DataLoader(
        trnSet, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        valSet, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )  # Use test_ds

    return VaeDatasets(train_loader, val_loader, test_loader, trnSet)


def get_data_if_not_exist():
    train_url, train_file_path = c.get_train_info()
    train_file_path = data_path + train_file_path
    print("Downloading data from:", train_url)
    print("Downloading data to:", train_file_path)
    ensure_directory_exists(train_file_path)
    if not os.path.exists(train_file_path):
        train_file_path = wget.download(train_url, train_file_path)

    test_url, test_file_path = c.get_test_info()
    test_file_path = data_path + test_file_path
    print("Downloading data from:", test_url)
    print("Downloading data to:", test_file_path)
    ensure_directory_exists(test_file_path)
    if not os.path.exists(test_file_path):
        test_file_path = wget.download(test_url, test_file_path)


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_anomaly_data() -> VaeDatasets:
    batch_size = c.get_batch_size()
    imageList, labelList = generate_image_info()
    print("input image shape : ", imageList.shape)
    print("input label shape : ", labelList.shape)
    ds = TensorDataset(T.tensor(imageList), T.tensor(labelList))
    length = [int(len(ds) * 0.7), int(len(ds) * 0.2)]
    length.append(len(ds) - sum(length))
    trnSet, valSet, tstSet = random_split(ds, length)
    # Data loaders
    train_loader = DataLoader(
        trnSet, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        valSet, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
    )
    test_loader = DataLoader(
        tstSet, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
    )
    return VaeDatasets(train_loader, val_loader, test_loader, trnSet)


def generate_image_info() -> Tuple[np.ndarray, np.ndarray]:
    data_list = [data_path + "/Anomaly_dataset.h5"]
    imageList = []
    labelList = []
    for file_path in data_list:
        print("Loading data from ", file_path)
        dataset = File(file_path, "r", libver="latest", swmr=True)
        FimageList = []
        FlabelList = []
        for _, group in dataset.items():
            for dName, data in group.items():
                if dName == "images":
                    FimageList.append(data)
                elif dName == "labels":
                    FlabelList.append(data)

        if len(FimageList) >= 2:
            # print("More than 2 gropus in File")
            image_concat = []
            for i in range(0, len(FimageList)):
                image_concat.append(FimageList[i][:])
            imageList.append(np.concatenate(image_concat))
            label_concat = []
            for i in range(0, len(FlabelList)):
                label_concat.append(FlabelList[i][:])
            labelList.append(np.concatenate(label_concat))
        else:
            imageList.append(FimageList[0][:])
            labelList.append(FlabelList[0][:])
    return np.concatenate(imageList), np.concatenate(labelList)


def main() -> None:
    train, validation, test, _ = get_anomaly_data()
    print(f"Training dataset size: {len(train)}")
    print(f"Validation dataset size: {len(validation)}")
    print(f"Testing dataset size: {len(test)}")
    train, validation, test, _ = get_house_data()
    print(f"Training dataset size: {len(train)}")
    print(f"Validation dataset size: {len(validation)}")
    print(f"Testing dataset size: {len(test)}")


if __name__ == "__main__":
    main()
