import numpy as np

class affine():
    def __init__(self, weight, bias):
        self.x = None
        self.weight = weight
        self.bias = bias
        
        self.d_w = None
        self.d_b = None

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
