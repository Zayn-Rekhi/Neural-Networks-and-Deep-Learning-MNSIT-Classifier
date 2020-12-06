import numpy as np


class Network():
    def __init__(self, shape):
        self.shape = shape
        self.length = len(shape)

        self.weights = [np.random.randn(y, x) \
                        for x, y in zip(self.shape[0:self.length-1], self.shape[1:self.length])]
        self.biases = [np.random.randn(y, 1) \
                        for y in self.shape[1:self.length]]



    def BGD(self, training_data, learning_rate, epochs):

        lenghtOfData = len(training_data)
        for epoch in range(0, epochs):
            nabla_weights = [np.zeros(weight.shape) for weight in self.weights]
            nabla_biases = [np.zeros(bias.shape) for bias in self.biases]

            for index, data in enumerate(training_data):
                weights, biases = self.backprop(data)
                nabla_weights = [w+nbw for w, nbw in zip(weights, nabla_weights)]
                nabla_biases = [b+nbb for b, nbb in zip(biases, nabla_biases)]

            self.weights = [w-(learning_rate/lenghtOfData)*nw \
                            for w, nw in zip(self.weights, nabla_weights)]
            self.biases = [b-(learning_rate/lenghtOfData)*nb \
                           for b, nb in zip(self.biases, nabla_biases)]

        # np.savez('weights_biases2.npz', weights=self.weights, biases=self.biases)


    def backprop(self, data):

        nabla_weights = [np.zeros(weight.shape) for weight in self.weights]
        nabla_biases = [np.zeros(bias.shape) for bias in self.biases]

        a = data[0]
        cost, zs = [data[0]], []
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, a) + bias
            zs.append(z)
            a = self.sigmoid(z)
            cost.append(a)

        error = (cost[-1]-data[1]) * self.sigmoid_prime(zs[-1])
        nabla_biases[-1] = error
        nabla_weights[-1] = np.dot(error, cost[-2].transpose())

        for l in range(2, self.length):
            error = np.dot(self.weights[-l+1].transpose(), error) * self.sigmoid_prime(zs[-l])
            nabla_biases[-l] = error
            nabla_weights[-l] = np.dot(error, cost[-l-1].transpose())

        return nabla_weights, nabla_biases

    def accuracy(self, training_data):
        running_average = []
        for data in training_data:
            output = self.feedforward(data[0])
            running_average.append((data[1], np.argmax(output)))
        print(running_average[500])
        return np.mean(running_average)

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        sig = self.sigmoid(x)
        return sig * (1-sig)