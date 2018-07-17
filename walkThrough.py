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
#from keras.models import model_from_json
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

path = r"C:\Users\Stin\Desktop\REU\Aye"
#r"C:\Python36\projects\kerasCode"
'''input()'''

while( not os.path.isdir(path)):
    print("ERROR: \'" + path + "\' is not a valid path!")
    print("Please enter the path name");
    path = input()

#load mnist dataset
# load and get data info 
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
numPixels = XTrain.shape[1] * XTrain.shape[2]
XTrain = XTrain.reshape(XTrain.shape[0], numPixels).astype('float32')
XTest = XTest.reshape(XTest.shape[0], numPixels).astype('float32')
#XTrain = XTrain / 255
#XTest = XTest / 255
yTrain = K.utils.to_categorical(yTrain)
yTest = K.utils.to_categorical(yTest)
numClasses = yTest.shape[1]

#set number of epochs
epochs = 1

#goes through ReLu folder and saves model weights
reluPath = os.path.join(path, 'ReLu')
for filename in os.listdir(reluPath):
    if filename.endswith(".txt"):
        print("Working on file " + filename)
        thePath = os.path.join(reluPath, filename)
        model = loadAndRunModel(thePath)
        
        #save the network weights
        model.save_weights(thePath[:-4] + "Weights.h5")

for folder, subs, files in os.walk(path):#get item for each directory in tree
    for filename in files:
        #make sure filetype is .txt
        if(filename.endswith('.txt')):
            print("Working on file " + filename)
            thePath = os.path.join(folder, filename)
            model = loadAndRunModel(thePath)

            '''#save model in model directory
            model_json = model.to_json()
            with open(thePath[:-4] + ".json", "w") as json_file:
                #save the model
                json_file.write(model_json)'''
