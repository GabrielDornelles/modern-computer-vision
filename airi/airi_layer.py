class AiriLayer:
    def __init__(self):
       pass
  
    def __call__(self, x):
        raise NotImplementedError("__call__ not implemented. This magicmethod should call your forward method.")
    
    def forward(self, x):
        raise NotImplementedError("forward not implemented. This is where you forward pass is written.")
    
    def backward(self, dout):
        raise NotImplementedError("backward not implemented. Without a backward step your block won't learn.")
    
    def update(self, **kwargs):
        raise NotImplementedError("update not implemented. You need to perform the weight update for this block.")

    def zero_grad(self, **kwargs):
        raise NotImplementedError("zero_grad not implemented. Gradients should be flushed every epoch.")