import numpy as np
import torch


class PILToTensor:
    def __call__(self, image, label):
        image = np.array(image).transpose((2, 0, 1))
        label = np.array(label)[np.newaxis, :, :]

        return torch.from_numpy(image), torch.from_numpy(label)
