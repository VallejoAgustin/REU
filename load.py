'''
mod: `load` -- Load Data
===========================================================
--  module :: load
   :platform: Windows10x64 [Need testing verification
              for other platforms]
   :synopsis: loads data from the mnist, cifar10,
              cifar100, fashionMNIST, reuters,
              imdb, and bostonHouston datasets,
              and normalizes or not, depending
              on the input passed-in.
              (defaults to normalize=True)
   :returns:  5 values, XTrain, yTrain, XTest,
              yTest, and numClasses
-- moduleauthor:: Agustin Vallejo
  
Requirements::
    1. Have numpy, keras, and tensorflow installed.
Todo::
    1. Ensure that the loading works for cifar10 and
       cifar100, fashionMNIST, imdb, reuters, and
       bostonHousing. [Have only created model for
       mnist to validate.]
Example::
    import load

    1. x_train, y_train, x_test, y_test, numClasses =
                        load.loadDataset("mnist")
    2. ... = load.loadDataset("fashionMNIST", normalize=False)
    3. ... = load.loadDataset("cifar10")
    4. ... = load.loadDataset("bostonHousing", normalize=False)
    Where '...' signifies the variables
'''
import numpy as np
from keras.utils import np_utils

def loadDataset(dataset, normalize=True):
    #seeding for reproducibility
    seed = 8
    np.random.seed(seed)

    #loads the data
    if dataset == "mnist":
        from keras.datasets import mnist
        (XTrain, yTrain), (XTest, yTest) = mnist.load_data()
        XTrain = XTrain.reshape(XTrain.shape[0], 1, 28, 28).astype('float32')
        XTest = XTest.reshape(XTest.shape[0], 1, 28, 28).astype('float32')
    elif dataset == "cifar10":
        from keras.datasets import cifar10
        (XTrain, yTrain), (XTest, yTest) = cifar10.load_data()
        XTrain=XTrain.astype('float32')
        XTest=XTest.astype('float32')
    elif dataset == "cifar100":
        from keras.datasets import cifar100
        (XTrain, yTrain), (XTest, yTest) = cifar100.load_data()
        XTrain=XTrain.astype('float32')
        XTest=XTest.astype('float32')
    elif dataset == "fashionMNIST":
        from keras.datasets import fashion_mnist
        (XTrain, yTrain), (XTest, yTest) = fashion_mnist.load_data()
    elif dataset == "imdb":
        from keras.datasets import imdb
        (XTrain, yTrain), (XTest, yTest) = imdb.load_data()
    elif dataset == "reuters":
        from keras.datasets import reuters
        (XTrain, yTrain), (XTest, yTest) = reuters.load_data()
    elif dataset == "bostonHousing":
        from keras.datasets import boston_housing
        (XTrain, yTrain), (XTest, yTest) = boston_housing.load_data()
    else:
        print("ERROR: Need functionality for " + dataset)
        return 0, 0, 0, 0, 0

    #one hot encode the outputs and set the number of classes
    yTrain = np_utils.to_categorical(yTrain)
    yTest = np_utils.to_categorical(yTest) 
    numClasses = yTest.shape[1]

    if normalize==True:
        XTrain /= 255
        XTest /= 255

    return XTrain, yTrain, XTest, yTest, numClasses
