from model_0 import *
from pathlib import Path 
import os

plot()

torch.manual_seed(20)
model_0 = LinearRegressionModel()

# print(model_0.state_dict())
loss_f = nn.L1Loss()
opt = torch.optim.SGD(model_0.parameters(), 
                      lr=0.01)

# plot_predictions(model_0(test_x).detach().numpy())
# print(model_0(test_x))

train_loss = []
test_loss = []
epochs_count = []
epochs = 300

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(train_x)

    loss = loss_f(y_pred, train_y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(test_x)
        test_loss_value = loss_f(test_pred, 
                            test_y.type(torch.float))
    
        train_loss.append(loss.detach().numpy())
        test_loss.append(test_loss_value.detach().numpy())
        if epoch%10==0:
            print(f'Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss_value}')


# print(model_0.parameters())
plot_predictions(model_0(test_x).detach().numpy())
plot_loss(range(epochs), train_loss, test_loss)

with torch.inference_mode():
    y_pred = model_0(test_x)


MODEL_PATH = Path('models')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
else:
    MODEL_NAME = 'model_0.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model_0.state_dict(),
            f=MODEL_SAVE_PATH)

loaded_0 = LinearRegressionModel()
loaded_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_0.eval()

with torch.inference_mode():
    print(y_pred == loaded_0(test_x))


