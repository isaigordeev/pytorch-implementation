import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from custom_setup import *


data_transform = transforms.Compose([

    transforms.Resize(size=(64,64)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor()
])


# test
def plot_transformed_images(image_paths, transform, n=3, seed=SEED):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    i = 0
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f'Original \nSize: {f.size}')
            ax[0].axis('off')

            trasformed_img = transform(f).permute(1, 2, 0)
            ax[1].imshow(trasformed_img)
            ax[1].set_title(f'Transformed \nSize: {trasformed_img.shape}')
            ax[1].axis('off')

            fig.suptitle(f'Class: {image_path.parent.stem}', fontsize=16)
        
            # plt.savefig(data_path / f'transform{i}.pdf')
            # i+=1 
    
# test_plot
# plot_transformed_images(image_path_list, 
#                         transform=data_transform,
#                         n=3)

# data acquiring
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)


print(f'Train data:\n{train_data}\nTest data:\n{test_data}')

class_names = train_data.classes
class_dict = train_data.class_to_idx

print(len(train_data), len(test_data))
# print(class_names)

img, label = train_data[0][0], train_data[0][1]

# print(f'{img}')
# print(f'{img.shape}')
# print(f'{img.dtype}')
# print(f'{label}')
# print(f'{type(label)}')

img_permute = img.permute(1, 2, 0)

plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.axis('off')
plt.title(class_names[label], fontsize=14)
# plt.savefig(data_path/'train_tuned_img.png')

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)

img, label = next(iter(train_dataloader))

# print({img.shape})
# print(label)
