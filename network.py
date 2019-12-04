# https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss #hàm tính Error cuối cùng của toàn mạng (1 scalar value)
        self.loss_prime = loss_prime #hàm tính đạo hàm của hàm Error ở trên (tính delta_Error/deltaY)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # FORWARD propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute LOSS (for display purpose only)
                err += self.loss(y_train[j], output)
                
                # Tính delta_Error/delta_Y (mse_prime or cross_entropy_prime) của network:
                error = self.loss_prime(y_train[j], output, self.layers[-1].input) # x_train[j])
                #layer[-1] là Activation, layers[-2] là Full Connect
                if j== 0:
                    print("delta_Error/delta_Y error = ", error)
                
                # BACKWARD propagation:
                for layer in reversed(self.layers):
                    #Tính error = delta_Error/delta_Y của lớp trước đó (hay delta_Error/delta_X của lớp hiện tại):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))