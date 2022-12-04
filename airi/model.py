from layers import *


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

    def forward(self, x, verbose=False, grad=True):
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