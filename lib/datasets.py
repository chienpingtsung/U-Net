from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageMaskFolder(Dataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super(ImageMaskFolder, self).__init__()

        self.image_path = Path(root).joinpath('images/')
        self.mask_path = Path(root).joinpath('masks/')

        image_stems = {p.stem for p in self.image_path.glob('*.png')}
        mask_stems = {p.stem for p in self.mask_path.glob('*.png')}
        assert not image_stems - mask_stems, f'missing mask of image: {image_stems - mask_stems}'
        assert not mask_stems - image_stems, f'missing image of mask: {mask_stems - image_stems}'

        self.stems = list(image_stems)

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        image = Image.open(self.image_path.joinpath(f'{self.stems[item]}.png'))
        mask = Image.open(self.mask_path.joinpath(f'{self.stems[item]}.png'))

        size = image.size

        if self.transforms:
            image, mask = self.transforms(image, mask)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask, size, self.stems[item]
