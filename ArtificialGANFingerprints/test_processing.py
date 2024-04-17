import PIL
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
import os
from torchvision.transforms import transforms
from tqdm import tqdm

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)

        return image, 0

    def __len__(self):
        return len(self.filenames)

file_path = '/home/kaden/Downloads/archive/img_align_celeba/img_align_celeba'

transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.ColorJitter(1)
            ]
        )

dataset = CustomImageFolder(file_path, transform)
dataloader = DataLoader(dataset, 32, False, )

save_path = '/home/kaden/Desktop/GAN_Watermarking/test/1'
os.makedirs(save_path, exist_ok=True)

for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

    torchvision.utils.save_image(data[0], os.path.join(save_path, f'{i}.png'))
