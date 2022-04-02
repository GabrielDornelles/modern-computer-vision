import numpy as np
# Those layers are meant to be used in notebook 4 - three layer convnet
# @Author: Gabriel Dornelles Monteiro
        
class Softmax:

    def __call__(self, x):
        return self.forward(x)

    def forward(self, scores):
        self.batch_size = scores.shape[0]
        scores -= np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(scores)
        self.softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) 
        return self.softmax_matrix
    
    def backward(self, y, loss="NLL"):
        if loss == "NLL":
            self.softmax_matrix[np.arange(self.batch_size) ,y] -= 1
            self.softmax_matrix /= self.batch_size
            return self.softmax_matrix
        raise NotImplementedError("Unsupported Loss Function") 
 

class LinearRelu:
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


class Linear:

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
    
