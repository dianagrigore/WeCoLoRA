from typing import Any, Callable, Optional, Tuple, Dict, Union, List

import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import pydicom
import torch
from hog import HOGLayerC

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


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


class RotateAndFlip():
    def __init__(self, img_size=(224, 224), cell_size=(16, 16)):
        self.img_size = img_size
        self.cell_size = cell_size
        self.coord_x_list = np.arange(0, self.img_size[0], self.cell_size[0])
        self.coord_y_list = np.arange(0, self.img_size[1], self.cell_size[1])

    def transform(self, x):
        labels = []

        x = x.permute((1, 2, 0))
        for coord_x in self.coord_x_list:
            for coord_y in self.coord_y_list:
                if np.random.random() < 0.4:
                    # Generate augmentation
                    aug = np.random.randint(1, 8)
                    labels.append(1) # aug

                    start_x = coord_x
                    end_x = start_x + self.cell_size[0]
                    start_y = coord_y
                    end_y = start_y + self.cell_size[1]
                    patch = x[start_y: end_y, start_x: end_x]

                    if aug == 1:  # flip
                        aux_patch = torch.fliplr(patch)
                    elif aug == 2:  # rot90
                        aux_patch = torch.rot90(patch)
                    elif aug == 3:  # rot90 + flip
                        aux_patch = torch.fliplr(torch.rot90(patch))
                    elif aug == 4:  # rot180
                        aux_patch = torch.rot90(patch, k=2)
                    elif aug == 5:  # rot180  + flip
                        aux_patch = torch.fliplr(torch.rot90(patch, k=2))
                    elif aug == 6:  # rot270
                        aux_patch = torch.rot90(patch, k=3)
                    elif aug == 7:  # rot270  + flip
                        aux_patch = torch.fliplr(torch.rot90(patch, k=3))

                    x[start_y: end_y, start_x: end_x] = aux_patch
                else:
                    labels.append(0)

        x = x.permute((2, 0, 1))

        return x, torch.Tensor(labels).long()

class MedicalDataset(datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            use_flip_rotate = False,
            use_hog_to_grayscale = False,
            target_transform: Optional[Callable] = None
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        self.samples = self.make_dataset()
        self.imgs = self.samples
        self.use_flip_rotate = use_flip_rotate
        self.use_hog_to_grayscale = use_hog_to_grayscale
        self.rotate_flip_op = RotateAndFlip()
        self.hog = HOGLayerC(nbins=9, pool=16)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        """
        {'paths': ['/media/lili/SSD2/datasets/ssl/medical/ct_colonography/manifest-sFI3R7DS3069120899390652954/CT COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0045/01-01-2000-1-Abdomen7MCVVIRTUALCOLONOSCOPY-47836/4.000000-Virt Colon  1.0 ST-47864/1-393.dcm', '/media/lili/SSD2/datasets/ssl/medical/ct_colonography/manifest-sFI3R7DS3069120899390652954/CT COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0045/01-01-2000-1-Abdomen7MCVVIRTUALCOLONOSCOPY-47836/4.000000-Virt Colon  1.0 ST-47864/1-394.dcm', '/media/lili/SSD2/datasets/ssl/medical/ct_colonography/manifest-sFI3R7DS3069120899390652954/CT COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0045/01-01-2000-1-Abdomen7MCVVIRTUALCOLONOSCOPY-47836/4.000000-Virt Colon  1.0 ST-47864/1-395.dcm'],
         'details': {'num_slices': 441, 'min_value': 0.0, 'max_value': 4095.0, 'mean_value': 612.8805273319858, 'std_value': 474.61859644882236}, 'identifier': '1.3.6.1.4.1.9328.50.4.0045_4.000000-Virt Colon  1.0 ST-47864'}
        """
        content = self.samples[index]
        sample = []
        for path_ in content['paths']:
            pixels = pydicom.dcmread(path_).pixel_array
            pixels[pixels == -2000] = 0
            sample.append(pixels)
        sample = np.float32(sample)

        sample = sample - content['details']['mean_value']
        sample = sample / (content['details']['std_value'] + 1e-8)

        sample = torch.Tensor(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        labels = -1
        if self.use_flip_rotate:
            sample, labels = self.rotate_flip_op.transform(sample)

        return sample, labels

    def __len__(self) -> int:
        return len(self.samples)

    def make_dataset(self) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample).

        """
        # a little bit hardcoded but it s fine.
        training_list = list(np.load('../data_processing/training_list_lidc.npy', allow_pickle=True)) + \
                        list(np.load('../data_processing/training_list_colon.npy', allow_pickle=True))

        return training_list
