"""
mod: `cMod` -- Create Models
===========================================================
--  module :: cMod
   :platform: Windows
   :synopsis: creates a keras model from given input file.
   :returns:  compiled model based on passed-in parameters
-- moduleauthor:: Agustin Vallejo
  
Requirements::
    1.  You will need to pip install keras,
        pip install tensorflow, and pip install
        numpy to run this code.
        #include downloads for keras
        #for tensorflow
        #for numpy
    2.  You will need to follow the following
        text file format:
        [For tuples do NOT include the parenthesis,
        and do NOT separate with spaces, ONLY commas.
        Columns are separated with tabs.]
        type	featureMaps/Neurons/Filters/activation	kernelSize	inputShape	strides
        use 'myM.txt' in repository as example
        e.g.:
        Conv2D	20	3,3	1,28,28	2,2 ini=random_uniform
        Activ	sigmoid
        Conv2D	10	2,2
        Activ	sigmoid
        Avg2DP	2,2
        Flatten
        Activ	sigmoid
        Dense	100
        Dense	10
    3.  The layer names are different for ease [Can easily be modified in-code].
        Convolutional2D -> Conv2D
        Activation -> Activ
        AveragePooling2D -> Avg2DP
        Flatten -> Flatten
        Dense -> Dense
    4.  To add a kernel initializer [currently only for Conv2D and Dense layers],
        create a new column at the end by tabbing and set the kernel initializer
        by declaring it through the notation: ini=[kernelIniHere].
        e.g: Dense    10    ini=lecun_uniform
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

#returns model from a text file of layers, has defaults to most used (in program)
def createModel(filename, lossF = 'categorical_crossentropy', opt = 'adam', met = ['accuracy']):
    #models we work on are all Sequential(), not Model()
    model = Sequential()

    #used to keep track of convolutional layers, 1st one has input size [other layers as first may cause error]
    conv = 0

    #place text file into list of lists
    with open(filename, 'r') as f:
        layers = []
        for line in f:
            cLine = []
            for word in line.split():
                cLine.append(word)
            layers.append(cLine)

    #add layers based off the text file by going through list of lists^
    for layer in layers:
        layer = np.array(layer)
        
        if(layer[0] == 'Conv2D'):
            #checks if kernel initializer set
            if 'ini=' in layer[len(layer)-1]:
                ini = layer[len(layer)-1]
                ini = ini[4:]
                print("We have: " + ini)
            else:
                ini = False
            
            conv += 1
            #create tuple for the kernel size
            x,y = layer[2].split(",")
            kS = [int(x), int(y)]
            kS = tuple(kS)
            
            #if input shape needs to be input
            if conv < 2:
                #create tuple for the input shape
                x,y,z = layer[3].split(",")
                iS = [int(x), int(y), int(z)]
                iS = tuple(iS)
            #if strides needs to be input and 1st conv 
            if len(layer) > 4 and conv == 1 and ini == False:
                x,y = layer[4].split(",")
                s = [int(x), int(y)]
                s = tuple(s)
                model.add(Conv2D(filters=int(layer[1]),kernel_size=kS,input_shape=iS, strides=s))
                continue
            if len(layer) > 4 and conv == 1 and ini != False:
                x,y = layer[4].split(",")
                s = [int(x), int(y)]
                s = tuple(s)
                model.add(Conv2D(filters=int(layer[1]),kernel_size=kS,input_shape=iS, strides=s, kernel_initializer=ini))
                continue
            #if strides needs to be input and 1st conv already occured
            if conv > 1 and len(layer) > 3 and ini == False:
                x,y = layer[3].split(",")
                s = [int(x), int(y)]
                s = tuple(s)
                model.add(Conv2D(filters=int(layer[1]),kernel_size=kS, strides=s))
                continue
            if conv > 1 and len(layer) > 4 and ini != False:
                x,y = layer[3].split(",")
                s = [int(x), int(y)]
                s = tuple(s)
                model.add(Conv2D(filters=int(layer[1]),kernel_size=kS, strides=s, kernel_initializer=ini))
                continue
            if ini != False:
                model.add(Conv2D(filters=int(layer[1]),kernel_size=kS, kernel_initializer=ini))
            else:
                model.add(Conv2D(filters=int(layer[1]),kernel_size=kS))
        elif(layer[0] == 'Activ'):
            model.add(Activation(layer[1]))
        elif(layer[0] == 'Avg2DP'):
            #create tuple for poolsize
            x,y = layer[1].split(",")
            pS = [int(x), int(y)]
            pS = tuple(pS)
            model.add(AveragePooling2D(pool_size=pS))
        elif(layer[0] == 'Flatten'):
            model.add(Flatten())
        elif(layer[0] == 'Dense'):
            #checks if kernel initializer set
            if 'ini=' in layer[len(layer)-1]:
                ini = layer[len(layer)-1]
                ini = ini[4:]
                model.add(Dense(int(layer[1]), kernel_initializer=ini))
            else:
                model.add(Dense(int(layer[1])))
        elif(layer[0] == 'Drop'):
            model.add(Dropout(float(layer[1])))
        else:
            print("[ERROR] Add functionality for: {s}".format(layer[0]))
            return 0
    model.compile(loss = lossF, optimizer = opt, metrics = met)

    return model

''' layers from regular model.add, Accuracies after 10 epochs: 89.67%, 89.66%, 89.66%
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 20, 13, 13)        200
_________________________________________________________________
activation_1 (Activation)    (None, 20, 13, 13)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 12, 12)        810
_________________________________________________________________
activation_2 (Activation)    (None, 10, 12, 12)        0
_________________________________________________________________
average_pooling2d_1 (Average (None, 10, 6, 6)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 360)               0
_________________________________________________________________
activation_3 (Activation)    (None, 360)               0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               36100
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
=================================================================
'''
'''layers from cMod: Accuracies: 92.3%, 92.29%, 92.33%,
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 20, 13, 13)        200
_________________________________________________________________
activation_1 (Activation)    (None, 20, 13, 13)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 12, 12)        810
_________________________________________________________________
activation_2 (Activation)    (None, 10, 12, 12)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 12, 12)        110
_________________________________________________________________
activation_3 (Activation)    (None, 10, 12, 12)        0
_________________________________________________________________
average_pooling2d_1 (Average (None, 10, 6, 6)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 360)               0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               36100
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
=================================================================
'''
