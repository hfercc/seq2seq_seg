from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
from glob import glob 
import numpy as np
import os.path as osp
import cv2
default_transforms = transforms.Compose([
    transforms.Resize((256,448)),
    transforms.ToTensor()
])

class YoutubeDataset(Dataset):

    def __init__(self, seq_len=5, split="train", base_dir='/raid/sunfangwen/wzh/youtube/', transforms = default_transforms):

        self.dir_list = open(split + "_dir.txt").readlines()
        self.base_dir = base_dir
        self.pathset = []
        self.transforms = transforms
        for a in self.dir_list:
            m, j = a.replace("\n", "").split(' ')
            full_path_j = osp.join(self.base_dir, j, "*.jpg")
            full_path_m = osp.join(self.base_dir, m, "*.png")
            m_path = sorted(glob(full_path_m))
            j_path = sorted(glob(full_path_j))
            for i in range(len(m_path) - seq_len + 1):
                self.pathset.append((j_path[i:i + seq_len], m_path[i:i + seq_len]))

    def __getitem__(self, idx):

        j_path, m_path = self.pathset[idx]
        j = []
        m = []
        for jp in j_path:
            img = Image.open(jp)
            img = self.transforms(img)
            j.append(img)
        for mp in m_path:
            img = cv2.imread(mp)

            img = cv2.resize(img, (256, 448))
            mask = np.zeros((2, 256, 448))
            img = img[:,:,0]
            img = np.transpose(img, (1, 0))
            mask[1][img == 1] = 1
            mask[0][img == 0] = 1
            mask = torch.from_numpy(mask)
            #/print(img)
            m.append(mask)
        j = torch.cat(j, 0)
        m = torch.cat(m, 0)
        return j, m

    def __len__(self):
        return len(self.pathset)


if __name__ == '__main__':
    dataset = YoutubeDataset()
    print(len(dataset))
    for (j,m) in dataset:
        print(j.shape)
        print(m.shape)
        break