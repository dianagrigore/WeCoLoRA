from typing import Any, Callable, Optional, Tuple, Dict, Union, List

import torchvision.datasets as datasets
from PIL import Image
import pdb
import numpy as np
import pydicom
import torch
import os
from sklearn.utils import shuffle
from tqdm import tqdm

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

IMG_EXTENSIONS = (".dcm")

def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img


def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


class RSNADataset(datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            is_train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )
        self.is_train = is_train
        self.num_classes = 6
        self.samples = self.make_dataset()
        self.imgs = self.samples
        self.data_dir = '/home/lili/dataset/rsna-intracranial-hemorrhage-detection/stage_2_train'

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        """
       {'paths': ['ID_85833caf5', 'ID_85833caf5', 'ID_ea89dfa0f'],
     'metadata': {'series': 'ID_b007382b3d',
      'mean': 402.77136,
      'std': 545.92224,
      'min': 0.0,
      'max': 3186.0,
      'num_images': 36},
     'label': {'any': 0,
      'epidural': 0,
      'intraparenchymal': 0,
      'intraventricular': 0,
      'subarachnoid': 0,
      'subdural': 0}}
       """
        content = self.samples[index]

        try:
            sample = bsb_window(pydicom.dcmread(os.path.join(self.data_dir, content['paths'][1] + '.dcm')))
        except:
            sample = np.zeros((512, 512, 3))

        sample = np.float32(sample)

        if self.transform is not None:
            sample = self.transform(image=sample)['image']

        label = torch.zeros(self.num_classes)
        labels = content['label']
        for idx, label_name in enumerate(['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']):
            label[idx] = labels[label_name]

        return sample, label

    def __len__(self) -> int:
        return len(self.samples)

    def make_dataset(self) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample).

        """
        # a little bit hardcoded but it s fine.
        if self.is_train:
            train_series = list(np.load('/home/lili/code/ssl/ssl-medical-sattelite/data_processing/train_series_rsna.npy', allow_pickle=True))
            # 21000
            train_series = shuffle(train_series, random_state=12)
            percent = 0.1
            num_of_kept_volumes = int(len(train_series) * percent)
            keep_volume_names = train_series[:num_of_kept_volumes]
            sample_list = list(np.load('/home/lili/code/ssl/ssl-medical-sattelite/data_processing/training_list_rsna.npy', allow_pickle=True))
            keep_sample_list = []
            for sample in tqdm(sample_list):
                if sample['metadata']['series'] in keep_volume_names:
                    keep_sample_list.append(sample)
            sample_list = keep_sample_list

        else:
            sample_list = list(np.load('/home/lili/code/ssl/ssl-medical-sattelite/data_processing/test_list_rsna.npy', allow_pickle=True))
        print(f'Loading {len(sample_list)} samples...')

        return sample_list
