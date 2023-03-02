import requests 
import zipfile
from pathlib import Path
from torch import nn 
import os 
import random
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt

SEED = 42
DEVICE = 'cpu' 
data_path = Path('models/vision/custom/')
image_path = data_path / 'pizza_steak_sushi'

if image_path.is_dir():
    print(f'{image_path} directory exists')
else: 
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print('Downloading pizza, steak, sushi data...')
        f.write(request.content)

    
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
        zip_ref.extractall(image_path)

# _____


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}')
        

walk_through_dir(data_path)


train_dir = image_path/ 'train'
test_dir = image_path/ 'test'


random.seed(SEED)

image_path_list = list(image_path.glob('*/*/*.jpg'))

random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem
# print(image_path_list, '\n')
# print(random_image_path, random_image_path.parent)

img = Image.open(random_image_path)

# print(f'rand path {random_image_path}')
# print(f'rand class {image_class}')
# print(f'height {img.height}')
# print(f'width {img.width}')

# img.show()

img_as_array = np.asarray(img)

plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f'class: {image_class}, shape: {img_as_array.shape}')
plt.axis(False)
# plt.savefig(data_path)