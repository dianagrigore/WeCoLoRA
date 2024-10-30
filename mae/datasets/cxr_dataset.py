import glob
import os
from typing import Callable, Optional

import torchvision.datasets as datasets
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np


class CXRDataset(datasets.VisionDataset):
    def __init__(self,
                 is_train: bool,
                 root: str = "/media/tonio/p2/datasets/CXR",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )
        self.is_train = is_train
        self.num_classes = 15
        self.class_list = ["Atelectasis", "Cardiomegaly", "Effusion",
                           "Infiltration", "Mass", "Nodule", "Pneumonia",
                           "Pneumothorax", "Consolidation", "Edema",
                           "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
                           "No Finding"]
        self.root = root
        self.images = list(glob.glob(os.path.join(root, "*", "*", "*.png")))
        self.samples, self.labels = self.make_dataset()
        print(len(self.samples))

    def __getitem__(self, item):
        img_path = self.samples[item]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        lbls = self.labels[item]
        return img, lbls

    def __len__(self):
        return len(self.samples)

    def make_dataset(self):
        samples = []
        labels = []
        trainval_list = f"{self.root}/train_val_list.txt" if self.is_train else f"{self.root}/test_list.txt"
        files = open(trainval_list,'r').read().split('\n')
        metadata = pd.read_csv(f"{self.root}/Data_Entry_2017_v2020.csv")
        metadata = dict(zip(metadata["Image Index"], metadata["Finding Labels"]))
        for img in tqdm(self.images):
            filename = img.split("/")[-1]
            if filename in files:
                samples.append(img)
                label = metadata[filename].split("|")
                sample_labels = [x in label for x in self.class_list]
                sample_labels = np.float32(sample_labels)
                assert sample_labels.sum() > 0 and sample_labels.sum() == len(label), f'{sample_labels}, {label}'

                labels.append(sample_labels)
        return samples, labels
