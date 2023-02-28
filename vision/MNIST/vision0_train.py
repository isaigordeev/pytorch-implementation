from vision0 import *


model_0 = FashionMNISTModel0(input_shape=784,
                             hidden_units=10,
                             output_shape=len(class_names))

print(model_0)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

torch.manual_seed(SEED)

train_time_cpu_start = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f'Epoch:{epoch}\n--------')

    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()

        y_pred = model_0(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 300 == 0:
            print(f'Batch:{batch * len(X)}/{len(train_dataloader.dataset)}')

    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0

    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:

            test_pred = model_0(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    
    print(f'\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n')

    train_time_cpu_end = timer()
    total_train_time = print_train_time(start=train_time_cpu_start,
                                        end=train_time_cpu_end,
                                        device='cpu')


torch.manual_seed(SEED)



model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)

print(model_0_results)


torch.save(obj=model_0.state_dict(),
            f=MODEL_SAVE_PATH)
