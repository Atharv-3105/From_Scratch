import os 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CaravanDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = os.listdir(image_dir) #It will get all the images present in the image_directory
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")) 
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #Converting to grayscale(Luminense)
        
        #Normalizing the mask value from 255.0 --> 1.0 because we will use sigmoid activation as we are doing binary segmentatioin also the loss will behave incorrectly for high value like 255
        mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image = image, mask= mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask
        
        
