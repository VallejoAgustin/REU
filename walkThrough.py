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
import os
import sys
import cMod
import load
from keras.models import model_from_json
import polynomials

print("Please enter the path name to the folders of polynomial text models");
print(r"An example pathname is: C:\Users\yourName\Desktop\foldereContainingPolys")

path = r"C:\Python36\projects\kerasCode"
#r"C:\Users\Stin\Desktop\REU\Ehsan Project"
'''input()'''

while( not os.path.isdir(path)):
    print("ERROR: \'" + path + "\' is not a valid path!")
    print("Please enter the path name");
    path = input()

#load mnist dataset
XTrain, yTrain, XTest, yTest, numClasses = load.loadDataset("mnist")

#set number of epochs
epochs = 1

for folder, subs, files in os.walk(path):#get item for each directory in tree
    for filename in files:
        #make sure filetype is .txt
        if(filename.endswith('.txt')):
            print("Working on file " + filename)
            thePath = os.path.join(folder, filename)
            model = cMod.createModel(thePath);

            #print summary
            print(model.summary())

            #fit model and print results
            model.fit(XTrain, yTrain, validation_data = (XTest, yTest), epochs = epochs, batch_size = 200)
            score = model.evaluate(XTest, yTest, verbose = 0)
            print("Accuracy: %.2f%%" % (score[1] * 100))

            #save model in model directory
            model_json = model.to_json()
            with open(thePath[:-4] + ".json", "w") as json_file:
                #save the model
                json_file.write(model_json)
            #save the network weights
            model.save_weights(thePath[:-4] + "Weights.h5")
