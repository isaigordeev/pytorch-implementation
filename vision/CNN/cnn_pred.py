from cnn import * 
import random 

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

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device:torch.device = 'cpu'):
    
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)


test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

print(test_samples[0].shape)


loaded_CNN_model = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names))

loaded_CNN_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

pred_probs = make_predictions(model=loaded_CNN_model, 
                              data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9,9))
rows = 3 
cols = 3

for i, sample in enumerate(test_samples):
    plt.subplot(rows, cols, i+1)

    plt.imshow(sample.squeeze(), cmap='gray')

    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]

    title_text = f'{pred_label} is {truth_label}'

    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')
    else:
        plt.title(title_text, fontsize=10, c='r')

    plt.axis(False);
    plt.savefig('./models/vision/CNN/predsCNN.pdf')



