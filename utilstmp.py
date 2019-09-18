import numpy as np
import pickle
import sys 
DataPath='/git/data'
sys.path.append(DataPath)
from CifarDataLoader import data_loading

try:
    import importlib
except ImportError:
    import imp as importlib

from chainer.datasets import get_mnist, get_cifar10
from chainer.cuda import get_device


GPU_ENABLED = False


def set_device(device_id):
    """
    Set the device (CPU or GPU) to be used.
    if device_id >= 0 the corresponding GPU is used, otherwise CPU is used.
    """
    if device_id < 0:
        # Use CPU
        return

    try:
        from cupy.cuda import Device
        from cupy.cuda.runtime import CUDARuntimeError
    except ImportError:
        print("Failed to import CuPy. Use CPU instead.")
        return

    try:
        Device(device_id).use()
    except CUDARuntimeError as e:
        print(e)
        return

    print("Device {} is in use".format(device_id))

    global GPU_ENABLED
    GPU_ENABLED = True

    # Reload the module to reflect the GPU status
    import pcanet
    importlib.reload(pcanet)


def gpu_enabled():
    return GPU_ENABLED


def reshape_dataset(train, test):
    def channels_last(X):
        X = np.swapaxes(X, 1, 2)
        X = np.swapaxes(X, 2, 3)
        return X

    X_train, y_train = train._datasets[0], train._datasets[1]
    X_test, y_test = test._datasets[0], test._datasets[1]
    X_train, X_test = channels_last(X_train), channels_last(X_test)
    return ((X_train, y_train), (X_test, y_test))


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def pick(train_set, test_set, n_train, n_test):
    images_train, y_train = train_set
    images_test, y_test = test_set
    train_set = images_train[:n_train], y_train[:n_train]
    test_set = images_test[:n_test], y_test[:n_test]
    return train_set, test_set


def load_cifar():
    train, test = data_loading(DataPath,'CIFAR10',1)
    return reshape_dataset(train, test)


def load_mnist(DataPath,dataset,batch_size):
    train_loader, test_loader = data_loading(DataPath,dataset,batch_size)
    #train=np.zeros(shape=(1,1,28,28))
   # test=np.zeros(shape=(1,1,28,28))

    
    return train_loader, test_loader


def concatenate_dicts(*dicts):
    """Concatenate multiple directories into one"""
    merged = []
    for d in dicts:
        merged += list(d.items())
    return dict(merged)
