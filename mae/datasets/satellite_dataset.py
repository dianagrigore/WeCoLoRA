import glob
import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None


class MillionAID(Dataset):
    def __init__(self, root, transforms=None):

        images = list(
            glob.glob(os.path.join(root, "test", "*", "*.jpg"))
        )
        self.transform = transforms
        self.files = images

        print('Creating MillionAID dataset with {} examples'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, -1


def create_NWPU_RESISC(path_to_dataset, args):
    dataset = torchvision.datasets.ImageFolder(root=path_to_dataset)
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.targets,
        stratify=dataset.targets,
        test_size=0.9,
        random_state=args.seed
    )

    dataset_train = Subset(dataset, train_indices)
    dataset_test = Subset(dataset, test_indices)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.3684, 0.3813, 0.3439], std=[0.1999, 0.1814, 0.1808])])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Resize((224, 224)),
                                                     torchvision.transforms.Normalize(mean=[0.3684, 0.3813, 0.3439],
                                                                                      std=[0.1999, 0.1814, 0.1808])])

    dataset_train.dataset.transform = transform_train
    dataset_test.dataset.transform = transform_test

    return dataset_train, dataset_test
