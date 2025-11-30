from torch.utils.data import DataLoader
from PIL import ImageOps
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import scipy.io as sio
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".gif", ".tiff"])

class fluorescence(Dataset):
    def __init__(self, data_folder, input_transforms=None, gray2rgb=False, rgb2gray=False):
        super(fluorescence, self).__init__()

        self.data = [os.path.join(data_folder, x) for x in os.listdir(data_folder) if is_image_file(x)]
        self.input_transforms = input_transforms
        self.grayscale2RGB = gray2rgb
        self.rgb2gray = rgb2gray


    def __getitem__(self, index):
        img = Image.open(self.data[index])

        if self.grayscale2RGB:
            img = img.convert('RGB')
        if self.rgb2gray:
            img = ImageOps.grayscale(img)

        if self.input_transforms:
            img = self.input_transforms(img)

        return img

    def __len__(self):
        return len(self.data)