import numpy as np
import os
import dill as pickle
from layers import *
from model import DemoConvNet


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def test_conv2d_forward():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    airi_conv = Conv2D(in_channels=3, num_filters=3, filter_size=4, stride=2, pad=1, custom_w=w, custom_b=b)

    out = airi_conv(x)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                            [-0.18387192, -0.2109216 ]],
                            [[ 0.21027089,  0.21661097],
                            [ 0.22847626,  0.23004637]],
                            [[ 0.50813986,  0.54309974],
                            [ 0.64082444,  0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                            [-1.19128892, -1.24695841]],
                            [[ 0.69108355,  0.66880383],
                            [ 0.59480972,  0.56776003]],
                            [[ 2.36270298,  2.36904306],
                            [ 2.38090835,  2.38247847]]]])
    print(f"    Relative error: {rel_error(out, correct_out)}")
    assert rel_error(out, correct_out) < 1e-3, "Airi Conv2d forward relative error is higher than 1e-3 compared to correct output."
    

def test_conv2d_backward():
    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2,)
    dout = np.random.randn(4, 2, 5, 5)

    airi_conv = Conv2D(in_channels=3, num_filters=3, filter_size=3, stride=1, pad=1, custom_w=w, custom_b=b)

    dx_num = eval_numerical_gradient_array(lambda x: airi_conv(x), x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: airi_conv(x), w, dout)
    db_num = eval_numerical_gradient_array(lambda b: airi_conv(x), b, dout)

    dx = airi_conv.backward(dout)
    dw = airi_conv.dW
    db = airi_conv.db

    assert rel_error(dx, dx_num) < 1e-3, "Dx relative error with numerical gradient is higher than 1e-3."
    assert rel_error(dw, dw_num) < 1e-3, "Dw relative error with numerical gradient is higher than 1e-3."
    assert rel_error(db, db_num) < 1e-3, "Db relative error with numerical gradient is higher than 1e-3."


def test_airi_model():
   
    with open(f'{os.path.abspath(os.getcwd())}/DemoConvNetv2.airi', 'rb') as f:
        model = pickle.load(f)
    
    x = np.load(f'{os.path.abspath(os.getcwd())}/utils/cifar10_truck.npy')
    x = x[None,...]
    y_hat = model.forward(x, verbose=True, grad=False)
    result = np.argmax(y_hat)

    assert y_hat.shape == (1, 10), f"DemoConvNet output shape should be (1, 10) but got {y_hat.shape}."
    assert result == 9, "Model answer was not idx 9 (truck), which should be if model loaded properly."
    assert y_hat[0][9] == 0.9999966846894813, f"Probability for idx 9 should be 0.9999966846894813 if model loaded properly, but it was {y_hat[0][9]}."

