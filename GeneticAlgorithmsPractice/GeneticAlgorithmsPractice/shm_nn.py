import numpy as np
import idx2numpy
import pickle

class FCLayer():

    # 0th column is bias, params -> output x input+1
    def __init__(self, input_params, output_params, is_input=False,
                 learn_rate=.001, momentum=0., activation='relu',
                 init_mean=0., init_var=.001, init_type='gaussian'):
        self.is_input = is_input
        if self.is_input == False:
            if init_type == 'gaussian':
                self.params = np.random.normal(init_mean, init_var, size=[output_params, input_params + 1])
            elif init_type == 'standard_normal':
                self.params = np.random.randn(output_params, input_params + 1) / np.sqrt(input_params)
        self.forward_pass_done = False
        self.backward_pass_done = False
        self.batch_size_changed = False
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.batch_size = 0
        self.activation_func = activation
        if activation == 'relu':
            self.activation = self.relu
            self.activation_backprop = self.d_relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_backprop = self.d_sigmoid
        elif activation == 'softmax':
            self.activation = self.softmax
            self.activation_backprop = self.d_cross_entropy_softmax
        elif activation == 'none':
            self.activation = self.no_activation
            self.activation_backprop = self.d_no_activation

    # batch_size x input+1 -> input, 0th column padded with 1s
    def forward(self, x):
        if self.is_input == True:
            self.forward_pass_done = True
            self.x = x.copy()
            self.y_activation = self.x.copy()
            return
        if self.forward_pass_done == False or x.shape[0] != self.batch_size:
            if x.shape[0] != self.batch_size:
                self.batch_size_changed = True
            self.x = np.zeros([x.shape[0], x.shape[1] + 1])
            self.x[:, 0] = 1.
            self.fwd_pass_done = True
            self.features = x.shape[1]
            self.batch_size = x.shape[0]
        self.x[:, 1:] = x.astype(np.float32)
        self.y = np.dot(self.x, self.params.T) # batch_size x output
        self.y_activation = self.activation(self.y)

    def relu(self, x):
        y = x.copy()
        y[y < 0.] = 0.
        return y

    def d_relu(self, y):
        dy = y.copy()
        dy[dy < 0.] = 0.
        dy[dy > 0.] = 1.
        return dy

    def sigmoid(self, x):
        y = 1. / (1. + np.exp(-x))
        return y

    def d_sigmoid(self, y):
        dy = y * (1. - y)
        return dy

    def softmax(self, x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        den = exps.sum(axis=1)
        den = np.tile(den, [exps.shape[1], 1]).T
        return exps / den

    def d_cross_entropy_softmax(self, truth): # truth -> one-hot vector
        derivative = self.y_activation - truth
        derivative /= derivative.shape[0]
        return derivative

    def no_activation(self, x):
        return x.copy()

    def d_no_activation(self, y):
        return y.copy()

    # derivatives -> batch_size x output
    def compute_gradients(self, derivatives):
        if self.is_input == True:
            self.batch_size_changed = False
            return
        if self.backward_pass_done == False:
            self.param_gradients = np.zeros_like(self.params)
            self.backward_pass_done = True
        if self.backward_pass_done == False or self.batch_size_changed == True:
            self.backprop_gradients = np.zeros_like([self.batch_size, self.features])
            self.batch_size_changed = False
        if self.activation_func != 'softmax':
            self.activation_derivatives = self.activation_backprop(self.y_activation)
            self.gradient_derivatives = derivatives * self.activation_derivatives
        else:
            self.gradient_derivatives = self.activation_backprop(derivatives)
        self.prev_param_gradients = self.param_gradients.copy()
        self.param_gradients = np.dot(self.x.T, self.gradient_derivatives)
        self.param_gradients = self.param_gradients.T #output x input+1
        self.backprop_gradients = np.dot(self.gradient_derivatives, self.params[:, 1:]) #batch_size x input

    def update_params(self, learn_rate=None, momentum=None):
        if self.is_input == False:
            if learn_rate == None:
                learn_rate = self.learn_rate
            if momentum == None:
                momentum = self.momentum
            self.params = self.params - learn_rate * self.param_gradients + momentum * self.prev_param_gradients


class FullyConnectedNeuralNet():

    def __init__(self, neuron_counts=[], load_path='', learn_rate=.001, momentum=0., 
                 activation='relu', init_mean=0., 
                 init_var=.001, init_type='standard_normal'):
        if load_path == '':
            self.layers = []
            self.input_layer = FCLayer(neuron_counts[0], neuron_counts[1], is_input=True, activation='none')
            self.layers.append(self.input_layer)
            self.num_layers = len(neuron_counts)
            for i in range(1, self.num_layers - 1):
                layer = FCLayer(neuron_counts[i - 1], neuron_counts[i], is_input=False,
                                learn_rate=learn_rate, momentum=momentum, activation='relu',
                                init_mean=init_mean, init_var=init_var, init_type=init_type)
                self.layers.append(layer)
            self.output_layer = FCLayer(neuron_counts[-2], neuron_counts[-1], is_input=False,
                                        learn_rate=learn_rate, momentum=momentum, activation='softmax')
            self.layers.append(self.output_layer)
        else:
            self.load(load_path)

    def feed_forward(self, x):
        self.layers[0].forward(x)
        for i in range(1, self.num_layers):
            self.layers[i].forward(self.layers[i - 1].y_activation)
        self.out = self.layers[-1].y_activation
        return self.out

    def get_backprop_gradients(self, y_one_hot):
        self.layers[-1].compute_gradients(y_one_hot)
        for i in range(self.num_layers - 2, 0, -1):
            self.layers[i].compute_gradients(self.layers[i + 1].backprop_gradients)

    def update_weights(self):
        for i in range(self.num_layers):
            self.layers[i].update_params()

    def train_step(self, x, y):
        self.feed_forward(x)
        self.get_backprop_gradients(y)
        self.update_weights()
        log_likelihood = -np.log((self.out * y).sum(axis=1))
        loss = np.mean(log_likelihood)
        return loss

    def save(self, path):
        pickle.dump(self.layers, open(path, 'wb'))

    def load(self, path):
        self.layers = pickle.load(open(path, 'rb'))
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.neuron_counts = [self.layers[0].x.shape[1]]
        for layer in self.layers[1:]:
            self.neuron_counts.append(layer.params.shape[0])
        self.num_layers = len(self.layers)

def to_one_hot(labels, classes):
    ret = np.zeros([labels.shape[0], classes]).astype(np.float)
    ret[range(labels.shape[0]), labels] = 1.
    return ret