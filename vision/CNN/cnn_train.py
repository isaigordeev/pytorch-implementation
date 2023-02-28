from cnn import *

torch.manual_seed(SEED)

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = 'cpu'
               ):
    train_loss, train_acc = 0,0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)


def test_step( model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = 'cpu'):
    test_loss, test_acc = 0,0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true = y,
                                    y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


model_1 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names))

print(model_1)



# images = torch.randn(size=(32,3,64,64))
# test_image = images[0]

# conv_layer = nn.Conv2d(in_channels = 3,
#                        out_channels=10,
#                        kernel_size=3,
#                        stride=1,
#                        padding=0)


# max_pool_layer = nn.MaxPool2d(kernel_size=2)

# print(test_image.shape)
# print(conv_layer(test_image).shape)

# test_conv = conv_layer(test_image)
# test_maxpool = max_pool_layer(test_conv)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

time_start = timer()
epochs = 3 

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n----------')
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)
    
time_end = timer()
total_time = print_train_time(start=time_start,
                              end= time_end,
                              device = 'cpu')

torch.save(obj=model_1.state_dict(),
            f=MODEL_SAVE_PATH)

