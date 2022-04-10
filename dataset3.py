import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


def make_dataset(root, is_train):
    if is_train:

        input = open(os.path.join(root, 'data/train_input.txt'))
        ground_t = open(os.path.join(root, 'data/train_gt.txt'))
        data_t = open(os.path.join(root, 'data/train_data.txt'))
        image = [(os.path.join(root, 'haze', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'gt', img_name.strip('\n'))) for img_name in
                 ground_t]
        data = [(os.path.join(root, 'sewer_resize_100_100', img_name.replace(".png", ".npy").strip('\n'))) for img_name in
              data_t]

        input.close()
        ground_t.close()
        data_t.close()


        return [[image[i], gt[i], data[i]]for i in range(len(image))]

    else:

        input = open(os.path.join(root, 'data/test_input.txt'))
        ground_t = open(os.path.join(root, 'data/test_gt.txt'))
        data_t = open(os.path.join(root, 'data/test_data.txt'))

        image = [(os.path.join(root, 'haze', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'gt', img_name.strip('\n'))) for img_name in
              ground_t]
        data = [(os.path.join(root, 'sewer_resize_100_100', img_name.replace(".png", ".npy").strip('\n'))) for img_name in
              data_t]

        input.close()
        ground_t.close()
        data_t.close()

        return [[image[i], gt[i], data[i]]for i in range(len(image))]



class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, data_path = self.imgs[index]
        #print(img_path)
        #print(gt_path)
        #print(data_path)
        img = Image.open(img_path)
        target = Image.open(gt_path)
        if self.triple_transform is not None:
            img, target, img_ = self.triple_transform(img, target, img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        data = np.load(data_path, allow_pickle=True).item()
        hough_space_label8 = data["hough_space_label8"].astype(np.float32)  
        hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0)
        gt_coords = data["coords"]

        return img, target, hough_space_label8, gt_coords

    def __len__(self):
        return len(self.imgs)
