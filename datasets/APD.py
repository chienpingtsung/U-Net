import logging
import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class APD202004v2crack(Dataset):
    """
    Cracks from APD202004v2 dataset.
    """

    def __init__(self, path):
        """You should organize your dataset folder as follow, and match each basename of images and labels.
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

        :param path: Path to the dataset.
        :param transform: Transform applied to data.
        """
        super(APD202004v2crack, self).__init__()

        self.image_path = os.path.join(path, 'images/')
        self.label_path = os.path.join(path, 'labels/')

        self.names = [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob(os.path.join(self.image_path, '*.bmp'))
        ]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.image_path, '{}.bmp'.format(self.names[item])))
        mask = Image.open(os.path.join(self.label_path, '{}.png'.format(self.names[item])))

        sample = {
            'image': np.array(image).transpose((2, 0, 1)),
            'mask': np.array(mask) // 255
        }

        return sample
