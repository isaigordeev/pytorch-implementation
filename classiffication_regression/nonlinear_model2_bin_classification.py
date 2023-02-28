import torch 
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary


n_samples = 1000
random_seed = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=random_seed)

# X = np.c_[X, X[:, 0]*X[:, 0],X[:, 1]*X[:, 1]]

circles = pd.DataFrame({'X1': X[:, 0],
                        'X2': X[:, 1],
                        'label': y})

# circles['X1^2': X[:, 2]]
# circles['X2^2': X[:, 3]]

# circles.columns = ['targets']

features = torch.tensor(np.column_stack([
                         circles['X1'].values,
                         circles['X2'].values]), dtype = torch.float)

# features = torch.cat((features, torch.tensor(circles['X1'].values).unsqueeze(dim = 1)), dim = 1)
# features = torch.cat((features, torch.tensor(circles['X2'].values).unsqueeze(dim = 1)),dim = 1)

labels = torch.tensor(circles['label'], dtype = torch.float)

print(features)
print(labels)

# print(labels)
# print(circles.label.value_counts())

# data init
# plt.scatter(x=circles['X1'],
#             y = circles['X2'],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.savefig('./models/model0_classification/init_data.pdf')

X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size=0.2,
                                                    random_state=random_seed)



class ClassificationNonLModel0(nn.Module):
    def __init__(self): 
        super().__init__()
        self.layer1 = nn.Linear(in_features=len(X_train[0]), out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(
            self.relu(self.layer2(
            self.relu(self.layer1(x)))))

torch.manual_seed(random_seed)
model_0 = ClassificationNonLModel0().to(device)
print(model_0)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)


def accuracy_fn(y, y_true):
    correct = torch.eq(y, y_true).sum().item()
    return (correct/len(y))*100

def sigmoid_round(y):
    if len(y.shape) > 1:
        return torch.round(torch.sigmoid(y).squeeze())
    return torch.round(torch.sigmoid(y))

# print(sigmoid_round(model_0(X_train[:5]).squeeze()))
# print(sigmoid_round(y_train[:5]))

# print(sigmoid_round(model_0(X_train[:5]))== sigmoid_round(y_train[:5]))

epochs = 1500


X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print(model_0(X_test)[:5])

torch.manual_seed(random_seed)
for epoch in range(epochs):
    model_0.train()


    y_raw = model_0(X_train).squeeze()
    y = torch.round(torch.sigmoid(y_raw))

    loss = loss_fn(y_raw,
                   y_train)
    acc = accuracy_fn(y = y,
                      y_true = y_train)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()


    model_0.eval()
    with torch.inference_mode(): 
        test_raw = model_0(X_test).squeeze()
        test = torch.round(torch.sigmoid(test_raw))

        loss_test = loss_fn(test_raw,
                       y_test)
        acc_test = accuracy_fn(y = test,
                          y_true = y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {loss_test:.5f}, Test acc: {acc_test:.2f}%")


# xx = np.column_stack([circles['X1'], circles['X2']])
# 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train, device)
plt.subplot(1, 2, 2)
plt.title("Test ")
plot_decision_boundary(model_0, X_test, y_test, device)
plt.savefig('./models/nonlinear_model0_classification/train_loss_seed:' + f'{random_seed}' + '.pdf')
plt.show()





