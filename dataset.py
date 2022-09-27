import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from scipy.ndimage.interpolation import zoom


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # Initialization function, read all images under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.npy'))


    def __getitem__(self, index):

        image_path = self.imgs_path[index]

        label_path = image_path.replace('image', 'label')

        image = np.load(image_path)
        label = np.load(label_path)
        wid, high = image.shape                    #Compressed resolution according to thesis requirements
        image = zoom(image, (384 / wid, 256 / high), order=3)
        label = zoom(label, (384 / wid, 256 / high), order=3)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label

    def __len__(self):

        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
