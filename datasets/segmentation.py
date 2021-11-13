import logging
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SegDataset(Dataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super(SegDataset, self).__init__()

        self.image_root = Path(root).joinpath('images/')
        self.label_root = Path(root).joinpath('labels/')

        image_stems = {p.stem for p in self.image_root.glob('*.png')}
        label_stems = {p.stem for p in self.label_root.glob('*.png')}
        assert not image_stems - label_stems, f'missing label of image: {image_stems - label_stems}'
        assert not label_stems - image_stems, f'missing image of label: {label_stems - image_stems}'

        self.stems = list(image_stems)

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        image = Image.open(self.image_root.joinpath(f'{self.stems[item]}.png'))
        label = Image.open(self.image_root.joinpath(f'{self.stems[item]}.png'))

        if self.transforms:
            image, label = self.transforms(image, label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
