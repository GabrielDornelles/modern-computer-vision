import numpy as np
import matplotlib.pyplot as plt
from layers import *
from im2col_cython import col2im_cython, im2col_cython
from torchvision.datasets import CIFAR10

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

class ThreeLayerConvNet:
    def __init__(self) -> None:
        # Build the model here as a dictionary, weights are differentiable blocks
        self.reg = 0.001
        self.lr = 0.001

        self.model = [
            Conv2D(in_channels=3, num_filters=6,filter_size=5, stride=1, pad=0, std=1e-3),
            Relu(),
            Conv2D(in_channels=6, num_filters=16, filter_size=5, stride=1, pad=0, std=1e-3),
            Relu(),
            Flatten(),
            Linear(input_size=9216, hidden_size=120, reg=self.reg, std=1e-3),
            Relu(),
            Linear(input_size=120, hidden_size=84, reg=self.reg, std=1e-3),
            Relu(),
            Linear(input_size=84, hidden_size=10, reg=self.reg, std=1e-3),
            Softmax()
        ]

    def forward(self, x, verbose= False):
        self.batch_size = x.shape[0]
        #print(x.shape)
        x = x.transpose(0,3,1,2) # Reshape to N C H W 
        for layer in self.model:
            if verbose: print(f"Forwarding Layer: {layer}, x shape: {x.shape}")
            x = layer.forward(x)
           
        return x
    
    def backward(self, y):
        loss = self.model[-1].NLLloss(y)
        dout = self.model[-1].backward(y)
        for layer in reversed(self.model[:-1]):
            dout = layer.backward(dout)
            layer.update(lr=self.lr)
        return loss

if __name__ == "__main__":
    model = ThreeLayerConvNet()
    epoches = 30
    num_train = train_images.shape[0]
    batch_size = 256
    X = train_images
    y = train_targets
    for i in range(epoches):
        batch_indices = np.random.choice(num_train, batch_size)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        model.forward(X_batch, verbose=False)
        loss = model.backward(y_batch)
        if i%1==0:
            print(f"Loss at epoch {i}: {loss}")
