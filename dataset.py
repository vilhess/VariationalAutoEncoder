from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os
from PIL import Image

class CelebA(Dataset):
    
    def __init__(self, path, transform=ToTensor()):
        self.path = path
        self.files = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_name = self.files[idx]
        tensor_image = self.transform(Image.open(os.path.join(self.path, image_name)))
        return tensor_image