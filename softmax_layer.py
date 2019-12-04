# https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65

from layer import Layer

import numpy as np

# inherit from base class Layer
class SoftmaxLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    #def __init__(self, input_size, output_size):
    def __init__(self):
        #Input X la chi so hang
        #Noron la chi so cot
        pass
        #self.weights = np.random.rand(input_size, output_size) - 0.5
        #self.bias = np.random.rand(1, output_size) - 0.5
        
    def softmax(self, x):
       e = np.exp(x)
       return e / np.sum(e, axis=1)
    
    def softmax_grad(self, s):
      #Đạo hàm riêng của hàm softmax theo xi
      # s là kết quả của hàm softmax theo xi
      # s là 1 vector, i, j là chỉ số của s
      
      """
       ∂ ij = s i . (1 − s j)  if i=j
       ∂ ij = − s i . s j      if i<>j
      """
      Jacobi = np.diag(s)
      len = s.size
      
      print("s = ", s)
      for i in range(len):
        for j in range(len):
          Jacobi[i][j] = (s[i] * (1 - s[i])) if (i == j) else (-s[i] * s[j])
      
      return Jacobi    
      
      
    def softmax_grad_vec(self, s):
      _s = s.reshape(-1, 1)
      
      return np.diagflat(_s) - np.dot(_s, _s.T)

  
    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        #self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self.softmax(self.input)
        return self.output

    # Computes dE/dW, dE/dB for a given output_error=dE/dY. 
    # Returns input_error = dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        return output_error
        
        #input_error = np.dot(output_error, self.weights.T)
        """
        input_error = self.softmax_grad(output_error)
        
        print("self.input.T = ", self.input.T)
        print("output_error = ", output_error)
        print("input_error = ", input_error)        
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
        """
        