import torch
from PIL import ImageFilter
from torchvision import transforms as tvtrans
from torchvision.transforms import functional


class ComposeTogether(tvtrans.Compose):
    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs


class ToTensorTogether(tvtrans.ToTensor):
    def __call__(self, *imgs):
        return (functional.to_tensor(img) for img in imgs)


class PILToTensorTogether(tvtrans.PILToTensor):
    def __call__(self, *imgs):
        return (functional.pil_to_tensor(img) for img in imgs)


class ResizeTogether(tvtrans.Resize):
    def forward(self, *imgs):
        return (functional.resize(img, self.size, self.interpolation, self.max_size, self.antialias) for img in imgs)


class PadTogether(tvtrans.Pad):
    def forward(self, *imgs):
        return (functional.pad(img, self.padding, self.fill, self.padding_mode) for img in imgs)


class RandomCropTogether(tvtrans.RandomCrop):
    def forward(self, *imgs):
        if self.padding is not None:
            imgs = (functional.pad(img, self.padding, self.fill, self.padding_mode) for img in imgs)

        width, height = functional.get_image_size(imgs[0])

        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            imgs = (functional.pad(img, padding, self.fill, self.padding_mode) for img in imgs)

        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            imgs = (functional.pad(img, padding, self.fill, self.padding_mode) for img in imgs)

        i, j, h, w = self.get_params(imgs[0], self.size)

        return (functional.crop(img, i, j, h, w) for img in imgs)


class RandomHorizontalFlipTogether(tvtrans.RandomHorizontalFlip):
    def forward(self, *imgs):
        if torch.rand(1) < self.p:
            return (functional.hflip(img) for img in imgs)
        return imgs


class RandomVerticalFlipTogether(tvtrans.RandomVerticalFlip):
    def forward(self, *imgs):
        if torch.rand(1) < self.p:
            return (functional.vflip(img) for img in imgs)
        return imgs


class RandomRotationTogether(tvtrans.RandomRotation):
    def forward(self, *imgs):
        angle = self.get_params(self.degrees)

        return (functional.rotate(img, angle, self.resample, self.expand) for img in imgs)


class Dilation:
    def __init__(self, size=3):
        self.size = size

    def __call__(self, *imgs):
        return (img.filter(ImageFilter.MaxFilter(self.size)) for img in imgs)
