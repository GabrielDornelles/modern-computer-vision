class AiriLayer:
    def __init__(self):
       pass
  
    def __call__(self, x):
        self.forward(x)
    
    def forward(self, x):
        pass
    
    def backward(self, dout):
        pass
    
    def update(self, **kwargs):
        pass

    def zero_grad(self, **kwargs):
        pass