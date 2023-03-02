from tqdm.auto import tqdm 
from cnn import * 
import torchmetrics
import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

loaded_CNN_model = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names))

loaded_CNN_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

y_preds = []
loaded_CNN_model.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making prediction"):
        # X, y = X.to('cpu'), y.to('cpu') 
        y_logit = loaded_CNN_model(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred)

y_pred_tensor = torch.cat(y_preds)

print(y_pred_tensor)
print(y_pred_tensor.shape)
# try:
#     import torchmetrics, mlxtend
#     print(f'mlxtend version: {mlxtend.__version__}')
#     assert int(mlxtend.__version__.split('.')[1]) >= 19
# except:
#     !pip install -q torchmetrics -U mlxtend 
#     import torchmetrics, mlxtend


confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds = y_pred_tensor,
                         target=test_data.targets)


fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(9,7)
)

plt.savefig('./models/vision/CNN/CNN_matrix.pdf')
# plt.show()
