# AIRI

AIRI is a short library implemented by me for educational purposes.

It contains vision AI blocks (Linear, ReLU, Softmax, Conv, flatten) with an AutoGrad like Api. Its built on top of numpy.

ConvNet example:
```py
class DemoConvNet:
    def __init__(self):
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
        x = x.transpose(0,3,1,2) # Reshape to N C H W 
        for layer in self.model:
            x = layer.forward(x)
        return x
    
    def backward(self, y):
        loss = self.model[-1].NLLloss(y)
        dout = self.model[-1].backward(y)
        for layer in reversed(self.model[:-1]):
            dout = layer.backward(dout)
            layer.update(lr=self.lr)
        return loss
```

To install Cython to perform Conv forward and backward (from cs231n) run the following command:

```sh
python3 setup.py build_ext --inplace
```
