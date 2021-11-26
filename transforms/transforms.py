import numpy as np
import torch


class PILToTensor:
    def __call__(self, image, label1, label2):
        image = np.array(image).transpose((2, 0, 1))
        label1 = np.array(label1)[np.newaxis, :, :]
        label2 = np.array(label2)[np.newaxis, :, :]

        return torch.from_numpy(image), torch.from_numpy(label1), torch.from_numpy(label2)
