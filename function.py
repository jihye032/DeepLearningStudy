import numpy as np

class sigmoid:
    def __init__(self):
        self.input = None
        self.out = None

    def forward(self, x):
        delta = 1e-7
        self.input = x
        self.out = 1 / (1 + np.exp(-x + delta))
        return self.out
        

    def backward(self, dout):
        dx = dout * self.out(1.0 - self.out)
        return dx



class relu:
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, dout):
        return np.maximum(0, dout)


class step_function:
    def __init__(self):
        pass

    def forward(self, x):
        return np.array(x > 0, dtype = np.int)

    def backward(self, dout):
        return np.array(dout > 0, dtype = np.int)



#여기는 다른 방식으로 진행할꺼임
def softmax(x):
    pass

def mean_square_error():
    pass

def cross_entropy_error(y, t):
    pass

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

# x = [x1 x2] -> y
# t = [0 1], [1,0]

if __name__ == '__main__':
    arr = np.array([2., 3.])
    y = step_function(input[i])
    print(y)
