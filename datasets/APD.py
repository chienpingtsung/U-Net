import logging
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        """
        The directory should be organized as following tree, and match each stem of images and labels.
        .
        ├── images
        │   ├── image0.bmp
        │   ├── image1.bmp
        │   ├── image2.bmp
        │   └── ...
        └── labels
            ├── image0.png
            ├── image1.png
            ├── image2.png
            └── ...

        :param root: Path to the dataset.
        :param transform: Transforms for data augmentation.
        """
        super(ImageFolder, self).__init__()

        self.image_path = Path(root).joinpath('images/')
        self.label_path = Path(root).joinpath('labels/')

        self.stems = [p.stem
                      for p in self.image_path.glob('*.bmp')]

        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        image = Image.open(self.image_path.joinpath(f'{self.stems[item]}.bmp'))
        label = Image.open(self.label_path.joinpath(f'{self.stems[item]}.png'))

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


class WNetDataset(Dataset):
    def __init__(self, images_root, seg_labels_root, edge_labels_root, transform=None):
        super(WNetDataset, self).__init__()

        self.images_root = Path(images_root)
        self.seg_labels_root = Path(seg_labels_root)
        self.edge_labels_root = Path(edge_labels_root)

        stems = {p.stem for p in self.images_root.glob('*.bmp')} & {p.stem for p in self.edge_labels_root.glob('*.png')}
        self.stems = list(stems)

        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        image = Image.open(self.images_root.joinpath(f'{self.stems[item]}.bmp'))
        seg_label = Image.open(self.seg_labels_root.joinpath(f'{self.stems[item]}.png'))
        edge_label = Image.open(self.edge_labels_root.joinpath(f'{self.stems[item]}.png'))

        if self.transform:
            image, seg_label, edge_label = self.transform(image, seg_label, edge_label)

        return image, seg_label, edge_label
