import numpy as np
import math
from im2col_cython import col2im_cython, im2col_cython
from airi_layer import AiriLayer

# Those layers are meant to be used in from notebook 4 beyond
# @Author: Gabriel Dornelles Monteiro

class Softmax(AiriLayer):

    def __init__(self, loss: str = "NLL"):
        self.loss_function = loss
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, scores, grad = True):
        self.batch_size = scores.shape[0]
        scores -= np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(scores)
        softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) 
        if grad:
            self.softmax_matrix = softmax_matrix
        return softmax_matrix
    
    def NLLloss(self, y):
        loss = np.sum(-np.log(self.softmax_matrix[np.arange(self.batch_size), y]))
        loss /= self.batch_size
        return loss
    
    def backward(self, y):
        if self.loss_function == "NLL":
            self.softmax_matrix[np.arange(self.batch_size) ,y] -= 1
            self.softmax_matrix /= self.batch_size
            return self.softmax_matrix
        raise NotImplementedError("Unsupported Loss Function")
    
    def zero_grad(self):
        self.softmax_matrix = None
 

class Flatten(AiriLayer):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, grad = True):
        if grad:
            self.old_shape = x.shape
        return np.reshape(x, (x.shape[0], -1))
    
    def backward(self, dout):
        return np.reshape(dout, self.old_shape)
    
    def update(self, **kwargs):
        pass
    
    def zero_grad(self):
        self.old_shape = None


class Conv2D(AiriLayer):

    def __init__(self, in_channels, num_filters, filter_size, stride, pad, reg=0.0, custom_w = None, custom_b = None):
        self.config = None
        self.config_b = None
        self.reg = reg
        self.stride = stride
        self.pad = pad
        if (custom_w is not None) and (custom_b is not None):
            self.conv = custom_w
            self.bias = custom_b
        else:
            # Pytorch like conv init: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L40
            k = in_channels * np.prod((filter_size, filter_size))
            unif = np.random.uniform(-1/math.sqrt(k), 1/math.sqrt(k))
            self.conv = np.random.randn(num_filters, in_channels, filter_size, filter_size) * unif
            self.bias = np.zeros(num_filters)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x, grad = True):
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

        x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
        out = out.transpose(3, 0, 1, 2)

        if grad:
            self.cache = (x, w, b, x_cols)
        return out

    def backward(self, dout):
        x, w, b, x_cols = self.cache
        stride, pad = self.stride, self.pad

        self.db = np.sum(dout, axis=(0, 2, 3))

        num_filters, _, filter_height, filter_width = w.shape 
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.dW = dout_reshaped.dot(x_cols.T).reshape(w.shape)
        self.dW += self.reg * 2 * self.conv 

        dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
        dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                        filter_height, filter_width, pad, stride)

        return dx  
    
    def update(self):
        self.bias, self.config_b = adam(self.bias, self.db, config=self.config_b)
        self.conv, self.config = adam(self.conv, self.dW, config=self.config)
    
    def zero_grad(self):
        self.dW = None
        self.dB = None
        self.cache = None


class Relu(AiriLayer):

    def __init__(self) -> None:
        self.local_derivative = lambda x: x > 0
  
    def __call__(self, x):
        self.forward(x)
    
    def forward(self, x, grad = True):
        if grad:
            self.x = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        return dout * self.local_derivative(self.x)
    
    def update(self, **kwargs):
        pass
    
    def zero_grad(self):
        self.x = None


class LinearRelu(AiriLayer):
    """
    Linear + Relu activation block. Usable, but not updated.
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
    
    def zero_grad(self):
        self.dW = 0
        self.dB = 0


class Linear(AiriLayer):
    # TODO: Better initialization with std being like stdv = 1. / math.sqrt(self.weight.size(1))
    def __init__(self, input_size=3072, hidden_size=10, reg=1e3, bias = True):
        self.config = None
        self.config_b = None
        self.bias = bias
        std = 1./ math.sqrt(input_size)
        self.w =  np.random.randn(input_size, hidden_size) * std
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
        
    def update(self):
        self.b, self.config_b = adam(self.b, self.dB, config=self.config_b)
        self.w, self.config = adam(self.w, self.dW, config=self.config)
    
    def zero_grad(self):
        self.dW = None
        self.dB = None
        self.x = None


def adam(w, dw, config=None):
    """
    ADAptative Moment estimtion
    From cs231n, source: https://github.com/jariasf/CS231n/blob/master/assignment2/cs231n/optim.py
    my notes: https://github.com/GabrielDornelles/softmax-classifier/blob/main/Cs231n%20Class%206%20-%20Backpropagation%20Techn.md
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-4)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
   
    eps, learning_rate = config['epsilon'], config['learning_rate']
    beta1, beta2 = config['beta1'], config['beta2']
    m, v, t = config['m'], config['v'], config['t']
    # Adam
    t = t + 1
    m = beta1 * m + (1 - beta1) * dw          # momentum
    mt = m / (1 - beta1**t)                   # bias correction
    v = beta2 * v + (1 - beta2) * (dw * dw)   # RMSprop
    vt = v / (1 - beta2**t)                   # bias correction
    next_w = w - learning_rate * mt / (np.sqrt(vt) + eps)
    # update values
    config['m'], config['v'], config['t'] = m, v, t

    return next_w, config