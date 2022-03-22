import os 
from torchvision import  transforms
from torch.utils.data import  Dataset

from PIL import Image


class Custom_Dataset(Dataset):
    def __init__(self, path, class_name):
        super().__init__()
        self.dims = (160, 160)      # As suggested, the network works best when the faces are at a fixed size of 160 X 160
        self.data_path = path
        self.name = class_name
        self.img_path = self._make_dataset()

    def _make_dataset(self):
        img_path = []
        for root, _, filenames in os.walk(os.path.join(self.data_path, self.name)):
            for i in filenames:
                if i.endswith(".jpg"):
                    img_path.append(os.path.join(root, i))
        return img_path

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        im = Image.open(img_path)
        return self.transforms(im)

    def transforms(self, img):
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(self.dims),
                                transforms.RandomGrayscale(p=0.1),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ColorJitter(brightness=.5, hue=.3),
                                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                # transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ])
        return t(img)

    def __len__(self):
        return len(self.img_path)

