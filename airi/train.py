import numpy as np
import matplotlib.pyplot as plt
from layers import *
from im2col_cython import col2im_cython, im2col_cython
from torchvision.datasets import CIFAR10
from rich.progress import track
import dill as pickle

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
# train_images shape: (50000, 3072)
print("Pre processed images")

class DemoConvNet:
    def __init__(self) -> None:
      
        self.reg = 0.0
        self.lr = 2.5e-4 #0.001

        # Dont optmize properly yet
        self.model = [
            Conv2D(in_channels=3, num_filters=6,filter_size=5, stride=1, pad=0),
            Relu(),
            Conv2D(in_channels=6, num_filters=16, filter_size=5, stride=1, pad=0),
            Relu(),
            Flatten(),
            Linear(input_size=9216, hidden_size=120, reg=self.reg, std=1e-3),
            Relu(),
            Linear(input_size=120, hidden_size=84, reg=self.reg, std=1e-3),
            Relu(),
            Linear(input_size=84, hidden_size=10, reg=self.reg, std=1e-3),
            Softmax()
        ]
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, verbose= False, grad = True):
        self.batch_size = x.shape[0]
       
        x = x.transpose(0,3,1,2) # Reshape to N C H W 
        for layer in self.model:
            if verbose: print(f"Forwarding Layer: {layer}, x shape: {x.shape}")
            x = layer.forward(x, grad = grad)
           
        return x
    
    def backward(self, y):
        loss = self.model[-1].NLLloss(y)
        dout = self.model[-1].backward(y)
        for layer in reversed(self.model[:-1]):
            dout = layer.backward(dout)
            layer.update(lr=self.lr)
            layer.zero_grad()
        return loss


def accuracy(model, x, y):
    x = model.forward(x, grad=False)
    y_pred = np.argmax(x, axis=1)
    return (y_pred == y).mean()

if __name__ == "__main__":

    ## Sanity check: Overfit small set of data: Working
    # model = DemoConvNet()
    # epoches = 250
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
    with open('DemoConvNet.airi', 'wb') as f:
        pickle.dump(model,f)
    epoches = 100
    num_train = train_images.shape[0]
    batch_size = 256
    X = train_images
    y = train_targets

    for i in range(epoches):
        # iterate the whole dataset in batches
        for batch in track(range(1, int(num_train/batch_size)), description="Training..."):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            model.forward(X_batch)
            loss = model.backward(y_batch)
        print(f"Loss for epoch {i}: {loss}")
        acc = accuracy(model, val_images, val_targets)
        print(f"Val acc for epoch {i}: {acc}")
    # Save model
    with open('DemoConvNet.airi', 'wb') as f:
        pickle.dump(model,f)
    
    # Load
    # with open('DemoConvNet.airi', 'rb') as f:
    #   y = pickle.load(f)
