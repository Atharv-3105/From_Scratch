from PIL import Image
img_path = "./data/train/0d3adbbc9a8b_01.jpg"
mask_path = "./data/train_masks/0d3adbbc9a8b_01_mask.gif"
print(Image.open(mask_path))