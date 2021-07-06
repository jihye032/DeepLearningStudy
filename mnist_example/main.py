import sys, os
sys.path.append(os.pardir)

import numpy as np
import pickle

from dataset.mnist import load_mnist
import layer

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, one_hot_label = True)
    
    return x_test, t_test

    
if __name__ == '__main__':
    
 

    
    # hyper parameters setting
    learning_rate = 1e-3
    hidden_input_size = 0
    hidden_output_size = 0

    print(x_train.shape, t_train.shape)
    print(x_test.shape, t_test.shape)

    batch_size = 100
    for epoch in range(1, 30):
        avg_cost = 0

        for i in range(int(math.ceil(len(x_train)/batch_size))):
