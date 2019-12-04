# https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65
from layer import Layer

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error = dE/dX for a given output_error = dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        # input_error = dE/dX = f'(X) * dE/dY
        
        if self.activation_prime is None:
            return output_error
        else:
            return self.activation_prime(self.input) * output_error