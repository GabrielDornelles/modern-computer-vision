# AIRI
-----
AIRI is a short library implemented by me for educational purposes.

It contains vision AI blocks (Linear, ReLU, Softmax, Conv2D, flatten) with an AutoGrad like Api. Its built on top of numpy.
# Setup
To install Cython to perform Conv forward and backward (from cs231n) run the following command:

```sh
python3 setup.py build_ext --inplace
```
# About 
- Linear and Conv2D layers are Kaiming initialized
- Adam is used as the optimizer
- It's modular, but some parts aren't (optimizers are not modular yet)

# Example

ConvNet implementation example:
```py
class DemoConvNet:
    def __init__(self) -> None:
      
        self.reg = 0.0
       
        self.model = [
            Conv2D(in_channels=3, num_filters=16,filter_size=5, stride=1, pad=0),
            Relu(),
            Conv2D(in_channels=16, num_filters=16, filter_size=5, stride=1, pad=0),
            Relu(),
            Flatten(),
            Linear(input_size=9216, hidden_size=120, reg=self.reg),
            Relu(),
            Linear(input_size=120, hidden_size=84, reg=self.reg),
            Relu(),
            Linear(input_size=84, hidden_size=10, reg=self.reg),
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
            layer.update()
            layer.zero_grad()
        return loss
```

Training code example:
```py
model = DemoConvNet()
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
```
You can find the code implementation for cifar-10 in train.py file.

Here is provided the weights for the above convnet that achieves **60%** validation accuracy on cifar-10 10k validation images: [Weights](https://drive.google.com/file/d/1bElQcUgwm-0lfaWSDM4zgD94lCLSHYOh/view?usp=sharing)
