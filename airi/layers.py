# Those layers are meant to be used in from notebook 4 beyond
# @Author: Gabriel Dornelles Monteiro
import numpy as np
from im2col_cython import col2im_cython, im2col_cython
from airi_layer import AiriLayer
#TODO: Weight Regularization for Conv2D, SGD+Momentum, Adam, MaxPool2D


class Softmax(AiriLayer):

    def __init__(self, loss: str = "NLL"):
        self.loss_function = loss
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, scores):
        self.batch_size = scores.shape[0]
        scores -= np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(scores)
        self.softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) 
        return self.softmax_matrix
    
    def NLLloss(self, y):
        loss = np.sum(-np.log(self.softmax_matrix[np.arange(self.batch_size), y]))
        loss /= self.batch_size
        # TODO: Regularization
        return loss
    
    def backward(self, y):
        if self.loss_function == "NLL":
            
            self.softmax_matrix[np.arange(self.batch_size) ,y] -= 1
            self.softmax_matrix /= self.batch_size
            return self.softmax_matrix
        raise NotImplementedError("Unsupported Loss Function") 
 

class Flatten(AiriLayer):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.old_shape = x.shape
        return np.reshape(x, (x.shape[0], -1))
    
    def backward(self, dout):
        return np.reshape(dout, self.old_shape)
    
    def update(self, **kwargs):
        pass


class Conv2D(AiriLayer):

    def __init__(self, in_channels, num_filters, filter_size, stride, pad, std):
        self.stride = stride
        self.pad = pad
        self.conv = std * np.random.randn(num_filters, in_channels, filter_size, filter_size)
        self.bias = np.zeros(num_filters)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        N, C, H, W = x.shape
        num_filters, _, filter_height, filter_width = self.conv.shape
        stride, pad = self.stride, self.pad
        w = self.conv
        b = self.bias

        # Check dimensions
        assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
        assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

        # Create output
        out_height = (H + 2 * pad - filter_height) // stride + 1
        out_width = (W + 2 * pad - filter_width) // stride + 1
        
        out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

        # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
        x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
        out = out.transpose(3, 0, 1, 2)

        self.cache = (x, w, b, x_cols)
        return out

    def backward(self, dout):
        x, w, b, x_cols = self.cache
        stride, pad = self.stride, self.pad

        self.db = np.sum(dout, axis=(0, 2, 3))

        num_filters, _, filter_height, filter_width = w.shape 
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.dW = dout_reshaped.dot(x_cols.T).reshape(w.shape)

        dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
        # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
        dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                        filter_height, filter_width, pad, stride)

        return dx #, dw, 
    
    def update(self, lr=1e-6):
        self.conv -= lr * self.dW
        self.bias -= lr * self.db


class Relu(AiriLayer):

    def __init__(self) -> None:
        self.local_derivative = lambda x: x > 0
  
    def __call__(self, x):
        self.forward(x)
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        return dout * self.local_derivative(self.x)
    
    def update(self, **kwargs):
        pass


class LinearRelu(AiriLayer):
    """
    Linear + Relu activation block
    """

    def __init__(self, std=1e-4, input_size=3072, hidden_size=10, reg=1e3, bias = True):
        self.bias = bias
        self.w = std * np.random.randn(input_size, hidden_size)
        self.b = np.zeros(hidden_size) if bias else None
        self.reg = reg

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x, grad = True):
        if grad:
            self.x = x
            self.fc = x@self.w + self.b
            return np.maximum(0, self.fc)
        return np.maximum(0, x@self.w + self.b)
        
    def backward(self, dout):
        dfc = dout * (self.fc > 0)
        self.dW = self.x.T@dfc
        self.dW += self.reg * 2 * self.w 
        self.dB = dfc.sum(axis=0) if self.bias else None
        return dout@self.w.T 
    
    def update(self, lr=1e-6):
        self.w -= lr * self.dW
        if self.bias:
            self.b -= lr * self.dB


class Linear(AiriLayer):

    def __init__(self, std=1e-4, input_size=3072, hidden_size=10, reg=1e3, bias = True):
        self.bias = bias
        self.w = std * np.random.randn(input_size, hidden_size)
        self.b = np.zeros(hidden_size) if bias else None
        self.reg = reg
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x, grad = True):
        if grad: 
            self.x = x
        return x@self.w + self.b
    
    def backward(self, dout):
        self.dW = self.x.T@dout
        self.dB = dout.sum(axis=0) if self.bias else None
        self.dW += self.reg * 2 * self.w 
        return dout@self.w.T 
        
    def update(self, lr=1e-6):
        self.w -= lr * self.dW
        if self.bias:
            self.b -= lr * self.dB
    
