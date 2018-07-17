'''
mod: `walkThrough` -- Walk through a directory's
                      subdirectories and reads text files
                      (used in conjunction with loadData and
                      cMod after already creating the files
                      with Julia Bristow's
                      generate_text_files.py) to run the model
                      and stores both the model and network
                      weights in the proper subdirectory.
===========================================================
--  module :: walkThroughPolys
   :platform: Windows
   :synopsis: finds all files in a dir and runs keras models
-- moduleauthor:: Agustin Vallejo
'''
# credit to: https://stackoverflow.com/questions/2212643/python-recursive-folder-read
# and https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python
# and https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# and https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
import os
import sys
import cMod
from keras.datasets import mnist
import keras as K
import functions

def loadAndRunModel(pathName):
    model = cMod.createModel(thePath);
    
    #print summary
    print(model.summary())

    #fit model and print results
    model.fit(XTrain, yTrain, validation_data = (XTest, yTest), epochs = epochs, batch_size = 200)
    score = model.evaluate(XTest, yTest, verbose = 0)
    print("Accuracy: %.2f%%" % (score[1] * 100))
    return model

print("Please enter the path name to the folders of polynomial text models");
print(r"An example pathname is: C:\Users\yourName\Desktop\foldereContainingPolys")

path = input()

while( not os.path.isdir(path)):
    print("ERROR: \'" + path + "\' is not a valid path!")
    print("Please enter the path name");
    path = input()

#load mnist dataset
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
numPixels = XTrain.shape[1] * XTrain.shape[2]
XTrain = XTrain.reshape(XTrain.shape[0], numPixels).astype('float32')
XTest = XTest.reshape(XTest.shape[0], numPixels).astype('float32')
yTrain = K.utils.to_categorical(yTrain)
yTest = K.utils.to_categorical(yTest)
numClasses = yTest.shape[1]

#set number of epochs
epochs = 50

ans = "ah"

print("Answer Y if this is the first time running the program")
while ans not in "YyNn":
    print("Do you want to reset the architecture weights? Y or N")
    ans = input()
    if ans == "N" or ans == "n":
        break;
    #goes through ReLu folder and saves model weights
    reluPath = os.path.join(path, 'ReLu')
    for filename in os.listdir(reluPath):
        if filename.endswith(".txt"):
            print("Working on file " + filename)
            thePath = os.path.join(reluPath, filename)
            model = loadAndRunModel(thePath)
            
            #save the network weights
            model.save_weights(thePath[:-4] + "Weights.h5")

epochs = 5

for folder, subs, files in os.walk(path):#get item for each directory in tree
    #open file to write results to
    outputFile = os.path.join(path, folder, "results.txt")

    f = open(outputFile, "a+")
    for filename in files:
        #make sure filetype is .txt and is not results
        if(filename.endswith('.txt')) and "arch" in filename:
            thePath = os.path.join(folder, filename)
            model = cMod.createModel(thePath);
    
            #print summary (i/o delay)
            #print(model.summary())

            #find relu weights file to load from
            arch = filename.split(r"_")
            weightPath = os.path.join(path, "ReLu", arch[0] + "_ReLUWeights.h5")
            model.load_weights(weightPath)

            #fit model and print results
            model.fit(XTrain, yTrain, validation_data = (XTest, yTest), epochs = epochs, batch_size = 200, verbose = 0)
            score = model.evaluate(XTest, yTest, verbose = 0)
            f.write(arch[1][:-4] + " accuracy: %.2f%%\n" % (score[1] * 100))
