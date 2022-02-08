import numpy as np

"Simple fully connected neural network architecture"

class Layers:
    def __init__(self):
        self.a = a
        self.a_out = a_out


class fully_connected(Layers):
    def __init__(self, in_n, out_n):
        self.w = np.random.randn(out_n, in_n)
        self.b = np.random.randn(out_n, 1)

    def forward_propagate(self, a):
        self.a = a
        a = np.dot(self.w, self.a) + self.b
        return a

    def backward_propagate(self, output_gradient, eta):
        weights_gradient = np.dot(output_gradient, self.a.T)
        self.w = self.w - eta*weights_gradient
        self.b = self.b - eta*output_gradient
        return np.dot(self.w.T, output_gradient)
    
class Activation(Layers):
    def __init__(self, activation, activation_prime=None):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagate(self, a):
        self.a = a
        return self.activation(self.a)

    def backward_propagate(self, output_gradient, eta):
        z = np.multiply(output_gradient, self.activation_prime(self.a))
        return z

class hyptan(Activation):
    def __init__(self):
        y = lambda x: np.tanh(x)
        y_ = lambda x: 1-np.tanh(x)**2
        super().__init__(y, y_)

class sigmoid(Activation):
    def __init__(self):
        y = lambda x: 1.0/(1+np.exp(-x))
        y_prime = lambda x: 1.0/np.power(1+np.exp(-x), 2)
        super().__init__(y, y_prime)

class LinearActivation(Activation):
    def __init__(self):
        y = lambda x: x
        y_prime = lambda x: 1
        super().__init__(y, y_prime)

class Swish(Activation):
    def __init__(self):
        y = lambda x: x*1.0/(1+np.exp(-x))
        y_prime = lambda x: x/(1+np.exp(-x))+(1.0/(1+np.exp(-x)))
        super().__init__(y, y_prime)

class Softmax(Activation):
    def __init__(self):
        softmax = lambda x: np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum()
        super().__init__(softmax)

def av_sqr_err(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def av_sqr_err_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/np.size(y_true)