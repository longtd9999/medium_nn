# https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65
import numpy as np

from network import Network
from fc_layer import FCLayer
from softmax_layer import SoftmaxLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, softmax, softmax_prime
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime

# TRAINING DATA
# input x_train shape is (4,1,2):
x_train = np.array([ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ])
#y_train =np.array([ [[0]], [[1]], [[1]], [[0]] ])
y_train = np.array([ [1,0], [0,1], [0,1], [1,0] ]) # 1 hot vector

# NETWORK
net = Network()
net.add(FCLayer(2, 3)) # input co 2 dimension [0,0], 3 neron 
# FCLayer sử dụng hàm activate là tanh, đạo hàm của tanh là tanh_prime:
net.add(ActivationLayer(tanh, tanh_prime))
#net.add(FCLayer(3, 1)) # input co 3 dimension, 1 neron
net.add(FCLayer(3, 2)) # input co 3 dimension, 2 neron
#net.add(ActivationLayer(tanh, tanh_prime))
net.add(ActivationLayer(softmax, softmax_prime))
"""
Thêm lớp ở trên thì bị lỗi sau:
Traceback (most recent call last):
  File "example_xor.py", line 36, in <module>
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
  File "E:\MyProg\Python\medium_nn\network.py", line 58, in fit
    error = layer.backward_propagation(error, learning_rate)
  File "E:\MyProg\Python\medium_nn\fc_layer.py", line 29, in backward_propagation
    weights_error = np.dot(self.input.T, output_error)
  File "<__array_function__ internals>", line 6, in dot
ValueError: shapes (3,1) and (2,2) not aligned: 1 (dim 1) != 2 (dim 0)
"""

#Neu them lop nay vao thi KQ sai ???
#net.add(SoftmaxLayer())

#Neu them lop nay vao thi KQ sai ???
#net.add(ActivationLayer(softmax, None))

# TRAIN
#net.use(mse, mse_prime)
net.use(cross_entropy, cross_entropy_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print("x_train= ",x_train)
print("y_train= ",y_train)
print("out predict= ",out)