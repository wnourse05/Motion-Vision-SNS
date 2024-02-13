import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class ClipDataset(Dataset):
    def __init__(self, root_dir, dtype=torch.float32, device='cpu'):
        self.root_dir = root_dir
        self.dtype=dtype
        self.classes = sorted(os.listdir(root_dir))
        self.classes = [float(self.classes[i]) for i in range(len(self.classes))]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.file_list = self._make_file_list()
        self.device = device

    def _make_file_list(self):
        file_list = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith('.p'):
                        file_list.append((os.path.join(class_dir, filename), class_dir))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, class_name = self.file_list[idx]
        # print(file_path)
        frames = torch.as_tensor(pickle.load(open(os.path.join(self.root_dir, file_path), 'rb')), dtype=self.dtype,
                                 device=self.device)

        label = class_name
        return frames, label

if __name__ == '__main__':
    train = ClipDataset('FlyWheelTrain')
    test = ClipDataset('FlyWheelTest')
    sample_img, sample_label = train[0]
    print(train.__len__())
    print(test.__len__())
    print(sample_img.shape, sample_label)

    train_dataloader = DataLoader(train, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=10, shuffle=False)

    imgs, labels = next(iter(train_dataloader))
    print(imgs.shape)