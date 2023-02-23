import torch 
from sklearn.datasets import make_circles
from torch import nn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


n_samples = 1000
random_seed = 42

X, y = make_circles(n_samples, 
                    noise=0.05,
                    random_state=random_seed)

X = np.c_[X, X[:, 0]*X[:, 0],X[:, 1]*X[:, 1]]

circles = pd.DataFrame({'X1': X[:, 0],
                        'X2': X[:, 1],
                        'X1^2': X[:, 2], 
                        'X2^2': X[:, 3], 
                        'label': y})

# circles.columns = ['targets']

features = torch.tensor(np.column_stack([
                         circles['X1'].values,
                         circles['X2'].values,
                         circles['X1^2'].values,
                         circles['X2^2'].values]), dtype = torch.float)

labels = torch.tensor(circles['label'], dtype = torch.float)

print(features)
print(labels)
print(circles.label.value_counts())

# data init
# plt.scatter(x=circles['X1'],
#             y = circles['X2'],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.savefig('./models/model0_classification/init_data.pdf')





