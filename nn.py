import numpy as np
import random

class NeuralNetwork:
    def __init__(self, sizes):
        self.layer_sizes = sizes
        self.num_layers = len(sizes)
        self.weights = []
        self.biases = []
        for y, x in zip(sizes[1:], sizes[:-1]):
            self.weights.append(np.random.randn(y, x))
            self.biases.append(np.random.randn(y, 1))
    
    def backprop(self, x, y):
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]
        ac = x
        acs = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, ac)+b
            zs.append(z)
            ac = activate(z)
            acs.append(ac)
        d = cost_prime(acs[-1], y) * sigmoid_prime(zs[-1])
        nb[-1] = d
        nw[-1] = np.dot(d, acs[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            d = np.dot(self.weights[-l+1].transpose(), d) * sigmoid_prime(z)
            nb[-l] = d
            nw[-l] = np.dot(d, acs[-l-1].transpose())
        return (nb, nw)

    def train_network(self, train_data, train_count, eta, test_data):
        train_data = list(train_data)
        n = len(train_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        if test_data:
                print(f"Train Count {-1} : {self.evaluate(test_data)} / {n_test}")

        mini_batch_size = 30

        for i in range(train_count):
            random.shuffle(train_data)
            mini_batches = [
                train_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_network(mini_batch, eta)

            if test_data:
                print(f"Train Count {i} : {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Train Count {i} complete")

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = activate(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update_network(self, data, eta):
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]

        for x, y in data:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nb = [nb+dnb for nb, dnb in zip(nb, delta_nabla_b)]
            nw = [nw+dnw for nw, dnw in zip(nw, delta_nabla_w)]
        self.weights = [w-(eta/len(data))*nw for w, nw in zip(self.weights, nw)]
        self.biases = [b-(eta/len(data))*nb for b, nb in zip(self.biases, nb)]


def cost_prime(a, y):
    return 2*(a-y)

# currently using sigmoid as ac function
def activate(z):
    return sigmoid(z)

# ac functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return (sigmoid(z)**2)*np.exp(-z)

def relu(z):
	return np.maximum(0.0, z)