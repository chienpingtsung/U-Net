import logging
import math

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)


class Tile:
    """
    Process image as description from Figure 2 of
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, image_res, mask_res):
        """As description from paper, mask after forward propagation should be central region of original image,
        because convolutions cause losing verge information, so only the centre could represent complete information.

        :param image_res: Target image resolution after Tile operation.
        :param mask_res: Target mask resolution after Tile operation.
        """
        assert (image_res - mask_res) % 2 == 0, \
            logger.error("Gap between iamge_res and mask_res should be dividable by 2.")

        self.image_res = image_res
        self.mask_res = mask_res

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        """Call for functional using.

        :param image: Original image (H, W, C) for splitting.
        :param mask: (optional) Original mask (H, W) for splitting. None for prediction purpose.
        """
        if type(mask) is np.ndarray:
            assert image.shape[0:2] == mask.shape, \
                logger.error("Unmatched resolution of image and mask.")

        height, width, _ = image.shape  # (H, W, C)

        n_row = math.ceil(height / self.mask_res)
        n_col = math.ceil(width / self.mask_res)

        padded_height = n_row * self.mask_res
        padded_width = n_col * self.mask_res

        top = (padded_height - height) // 2
        bot = math.ceil((padded_height - height) / 2)
        lef = (padded_width - width) // 2
        rig = math.ceil((padded_width - width) / 2)

        cornicione = (self.image_res - self.mask_res) // 2

        image = np.pad(image, (
            (top + cornicione, bot + cornicione),
            (lef + cornicione, rig + cornicione),
            (0, 0)  # not pad channel
        ), mode='reflect')
        images = np.stack([
            image[top:top + self.image_res, lef:lef + self.image_res, :]
            for top in range(0, padded_height, self.mask_res)
            for lef in range(0, padded_width, self.mask_res)
        ])

        if type(mask) is np.ndarray:
            mask = np.pad(mask, (
                (top, bot),
                (lef, rig)
            ), mode='reflect')
            masks = np.stack([
                mask[top:top + self.mask_res, lef:lef + self.mask_res]
                for top in range(0, padded_height, self.mask_res)
                for lef in range(0, padded_width, self.mask_res)
            ])

            return images, masks  # for training and testing

        return images  # for predicting


class Dilate:
    """
    By practicing, we found that in some cases, dilating the labels of entities could help a lot especially tiny ones.
    """

    def __init__(self, size=3):
        assert 1 < size, \
            logger.error("Dilating operation requires a filter size greater than 0.")

        self.size = size

    def __call__(self, mask):
        return ndimage.maximum_filter(mask, size=self.size)


class Detile:
    def __call__(self, tiles, size, lef, top):
        """

        :param tiles: ndarray shape like (B, H, W)
        :param size: size like (W, H)
        :param lef:
        :param top:
        """
        im = Image.new('L', size)
        i = 0
        for t in range(-top, -top + size[1], tiles.shape[1]):
            for l in range(-lef, -lef + size[0], tiles.shape[2]):
                tile = Image.fromarray(tiles[i])
                im.paste(tile, (l, t))
                i += 1

        return im
