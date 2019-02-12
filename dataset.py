from torch.utils.data import Dataset, DataLoader
from glob import glob 
import os.path as osp

class YoutubeDataset(Dataset):

    def __init__(self, seq_len=5, split="train", base_dir='/raid/sunfangwen/wzh/youtube/'):

        self.dir_list = open(split + "_dir.txt").readlines()
        self.base_dir = base_dir
        self.pathset = []
        for a in self.dir_list:
            j, m = a.split(' ')
            full_path_j = osp.join(self.base_dir, j, "*.png")
            full_path_m = osp.join(self.base_dir, m, "*.png")
            m_path = sorted(glob(full_path_m))
            j_path = sorted(glob(full_path_j))
            for i in range(len(m_path) - seq_len + 1):
                self.pathset.append((j_path[i, i + seq_len], m_path[i, i + seq_len]))

    def __len__(self):
        return len(self.pathset)


if __name__ == '__main__':
    dataset = YoutubeDataset()
    print(len(dataset))