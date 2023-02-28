from vision0 import *

loaded_0 = FashionMNISTModel0(input_shape=784,
                             hidden_units=10,
                             output_shape=len(class_names))


loaded_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_0.eval()

plt.clf()
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(test_data), size=[1]).item()
    img, label = test_data[random_idx]
    with torch.inference_mode():
        pred = loaded_0(img).argmax(dim=1)
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[label] + ' is: ' + class_names[pred] )
    plt.axis(False);

plt.savefig('./models/vision/MNIST_vision/image_pick_set_prediction4_wout_last_ReLU.png')