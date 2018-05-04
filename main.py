import torch
import torch.nn as nn
import torch.optim as optim
import helper.datahelper as datahelper
from logistic_regression import LogisticRegression


if __name__ == '__main__':

    # hyper parameters
    epochs = 50
    image_size = 28
    batch_size = 100
    classes = 10
    lr = 0.001

    # get data
    train_data_loader = datahelper.get_mnist_train_data_loader(
        image_size=image_size,
        batch_size=batch_size
    )

    test_data_loader = datahelper.get_mnist_test_data_loader(
        image_size=image_size,
        batch_size=batch_size
    )

    # instantiate model
    model = LogisticRegression(image_size * image_size, classes)

    # instantiate loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (batch, labels) in enumerate(train_data_loader):

            batch = batch.view(batch_size, image_size * image_size)

            # empty the gradients of the optimizer
            optimizer.zero_grad()

            # forward pass
            output = model(batch)

            # compute loss
            loss = criterion(output, labels)

            # compute gradients using backpropagation
            loss.backward()

            # update parameters using optimizer.step()
            optimizer.step()

        # print loss
        print(
            'Epoch [%d/%d], Loss: %.4f'
            % (epoch + 1, epochs, loss.item())
        )

    correct = 0
    total = 0

    for batch, labels in test_data_loader:
        batch = batch.view(batch_size, image_size * image_size)
        output = model(batch)

        # get the max for each instance in the batch
        _, output = torch.max(output.data, 1)
        total += labels.size()[0]

        correct += torch.sum(torch.eq(labels, output))

    correct = correct.item()
    accuracy = correct / total * 100
    print('Accuracy:' + str(accuracy))
