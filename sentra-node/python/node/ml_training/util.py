import numpy as np

def loadMNISTDataset():
    # Load datasets
    data = np.load('/mnist.npz')
    x_train2, y_train2 = data['x_train'], data['y_train']
    x_test2, y_test2 = data['x_test'], data['y_test']
    return (x_train2,y_train2),(x_test2,y_test2)