import numpy as np
import matplotlib.pyplot as plt
from layers import *
from torchvision.datasets import CIFAR10
from rich.progress import track
import dill as pickle
from model import DemoConvNet


if __name__ == "__main__":

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Boilerplate do notebook anterior
    train_dataset = CIFAR10(root="./", download=True, train=True) 
    val_dataset = CIFAR10(root="./", download=True, train=False) 

    train_images = np.array([np.array(train_dataset[i][0]) for i in range(len(train_dataset))])
    train_targets = np.array([np.array(train_dataset[i][1]) for i in range(len(train_dataset))])

    val_images = np.array([np.array(val_dataset[i][0]) for i in range(len(val_dataset))])
    val_targets = np.array([np.array(val_dataset[i][1]) for i in range(len(val_dataset))])

    #transforma nossas imagens 32x32x3 em vetor linha 3072
    # train_images = np.reshape(train_images, (train_images.shape[0], -1))
    # val_images = np.reshape(val_images, (val_images.shape[0], -1))

    # média do array train_images no eixo 0, eixo onde os índices são as imagens 3072
    mean_image = np.mean(train_images, axis = 0)

    # mean_image é um array de floats, para operação fazer sentido nossas imagens precisam ser também.
    train_images = train_images.astype(float)
    train_images -= mean_image

    val_images = val_images.astype(float)
    val_images -= mean_image
    print("Pre-processed cifar-10 Images")

    def accuracy(model, x, y):
        x = model.forward(x, grad=False)
        y_pred = np.argmax(x, axis=1)
        return (y_pred == y).mean()
    # Sanity check: Overfit small set of data: Working
    #print("Sanity check")
    # model = DemoConvNet()
    # epoches = 60
    # num_train = 100 
    # batch_size = 50
    # X = train_images[:100]
    # y = train_targets[:100]
    # for i in range(epoches):
    #     batch_indices = np.random.choice(num_train, batch_size)
    #     X_batch = X[batch_indices]
    #     y_batch = y[batch_indices]
    #     model.forward(X_batch)
    #     loss = model.backward(y_batch)
    #     print(f"Loss for epoch {i}: {loss}")

    # Training

    model = DemoConvNet()

    # Load model or checkpoint
    # with open('DemoConvNet.airi', 'rb') as f:
    #     model = pickle.load(f)
    
    epoches = 20
    num_train = train_images.shape[0]
    batch_size = 128
    X = train_images
    y = train_targets
    best_acc = 0.0
    for i in range(epoches):
        # iterate the whole dataset in batches
        for batch in track(range(1, int(num_train/batch_size)), description="Training..."):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            model.forward(X_batch)
            loss = model.backward(y_batch)
        print(f"Loss for epoch {i}: {loss}")
        acc = accuracy(model, val_images[:1000], val_targets[:1000])
        print(f"Val acc for epoch {i}: {acc}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    # Save model
    with open('DemoConvNet.airi', 'wb') as f:
        pickle.dump(best_model,f)
    

