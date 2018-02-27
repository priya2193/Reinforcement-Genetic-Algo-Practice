import cv2
import numpy as np
import idx2numpy
import os
from sklearn.metrics import accuracy_score

from shm_nn import *

data_dir = 'C:\data\mnist'

def get_mnist_data(x_path, y_path):
    x_in = idx2numpy.convert_from_file(x_path)
    x_in = np.array([x.flatten() for x in x_in])
    y_in = idx2numpy.convert_from_file(y_path)
    return x_in / 255., y_in

def get_mnist_testset():
    return get_mnist_data(data_dir + os.sep + 't10k-images.idx3-ubyte', data_dir + os.sep + 't10k-labels.idx1-ubyte')

def get_mnist_trainset():
    return get_mnist_data(data_dir + os.sep + 'train-images.idx3-ubyte', data_dir + os.sep + 'train-labels.idx1-ubyte')

def show(img_cv2, res=(700, 700)):
    cv2.imshow('image', cv2.resize(img_cv2, res, cv2.INTER_CUBIC))
    cv2.waitKey()

def show_(img_cv2, res=(700, 700)):
    cv2.imshow('image', img_cv2)
    cv2.waitKey()


def load_and_test():
    x_test, y_test = get_mnist_testset()
    nn = FullyConnectedNeuralNet(load_path='mnist_model.nn')
    preds = nn.feed_forward(x_test).argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    print('Validation accuracy =', 100. * acc, '%')


def train_and_save():
    x_train, y_train = get_mnist_trainset()
    x_test, y_test = get_mnist_testset()
    y_train_onehot = to_one_hot(y_train, 10)
    nn = FullyConnectedNeuralNet([784, 64, 32, 10], learn_rate=.8, momentum=.009, init_type='gaussian', activation='sigmoid')
    batch_size = 128
    epochs = 20
    iters = 1
    test_iter_interval = 200
    steps = x_train.shape[0] // batch_size
    training_done = False
    for epoch in range(epochs):
        if training_done == True:
            break
        for step in range(steps):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size
            loss = nn.train_step(x_train[start_idx:end_idx], y_train_onehot[start_idx:end_idx])
            print('Epoch', epoch + 1, 'Batch', step + 1, 'Iters', iters, 'Loss =', loss)
            if iters % test_iter_interval == 0:
                preds = nn.feed_forward(x_test).argmax(axis=1)
                acc = accuracy_score(y_test, preds)
                print('Validation accuracy =', 100. * acc, '%')
                if acc > .95:
                    nn.save('mnist_model.nn')
                    training_done = True
            iters += 1

train_and_save()
#load_and_test()