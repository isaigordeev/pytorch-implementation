from torch import nn 
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from tqdm.auto import tqdm
from timeit import default_timer as timer 


import torch 
import torchvision 
import matplotlib.pyplot as plt 

MODEL_SAVE_PATH = 'models/vision/MNIST_vision/MNIST_vision.pth'


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss, acc = 0,0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
            
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {'model_name': model.__class__.__name__,
            'model_loss':loss.item(),
            'model_acc': acc}

SEED = 42 

train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None    
)


test_data = datasets.FashionMNIST(
    root='data',
    train='False',
    download=True,
    transform=ToTensor()
)

image, label = train_data[0]

# print(image)
# print(label)

# print(image.shape)
class_names = train_data.classes 
# print(train_data.classes)

print(f'shape: {image.shape}')
plt.imshow(image.squeeze(), cmap='gray')
plt.title(class_names[label])
# plt.savefig('./models/MNIST_vision/image_pick.png')


torch.manual_seed(SEED)
fig = plt.figure(figsize=(9,9))
rows, cols = 4,4

# plt.clf()
# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap='gray')
#     plt.title(class_names[label])
#     plt.axis(False);
# plt.savefig('./models/MNIST_vision/image_pick_set.png')

BATCH_SIZE = 32 

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# print(len(train_dataloader))


train_features_batch, train_label_batch = next(iter(train_dataloader))

# print(train_features_batch.shape)

torch.manual_seed(SEED)
# plt.clf()
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_label_batch[random_idx]
# plt.imshow(img.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.savefig('./models/MNIST_vision/image_pick_batch.png')


flatten_model = nn.Flatten()

x = train_features_batch[0]

output = flatten_model(x)

print(x.shape)
print(output)
print(output.shape)

class FashionMNISTModel0(nn.Module):
    def __init__(self, input_shape: int, hidden_units : int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer_stack(x)




