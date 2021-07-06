import numpy as np

class relu:
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, dout):
        return np.maximum(0, dout)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return sum_exp_a

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / y.shape[0]


class softmax_CEE():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class affine():
    def __init__(self, weight, bias):
        self.x = None
        self.weight = weight
        self.bias = bias
        
        self.d_w = None
        self.d_b = None
        
    '''
    # it's better for save memory and init weight, bias
    def __init__(self, input_size, output_size):
        self.W = np.randn(output_size,input_size)
        self.b = np.zeros(output_size)
        ...
    '''

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, dout):
        self.d_w = np.transpose(self.x) * dout
        self.d_b = dout
        return dout * np.transpose(self.weight)
    
    def update(self, lr):
        self.weight -= lr * self.d_w
        self.bias -= lr * self.d_b
