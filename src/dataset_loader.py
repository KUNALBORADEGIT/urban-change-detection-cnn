import os
import cv2
import torch
from torch.utils.data import Dataset

class LEVIRDataset(Dataset):

    def __init__(self, root_dir):

        self.dir_A = os.path.join(root_dir, "A")
        self.dir_B = os.path.join(root_dir, "B")
        self.dir_label = os.path.join(root_dir, "label")

        self.images = sorted(os.listdir(self.dir_A))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name)
        path_label = os.path.join(self.dir_label, img_name)

        imgA = cv2.imread(path_A)
        imgB = cv2.imread(path_B)
        label = cv2.imread(path_label, 0)

        imgA = cv2.resize(imgA,(256,256))
        imgB = cv2.resize(imgB,(256,256))
        label = cv2.resize(label,(256,256))

        imgA = imgA / 255.0
        imgB = imgB / 255.0

        imgA = torch.tensor(imgA).permute(2,0,1).float()
        imgB = torch.tensor(imgB).permute(2,0,1).float()
        label = torch.tensor(label).unsqueeze(0).float()

        return imgA, imgB, label