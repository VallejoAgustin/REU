import load
import cMod

#load mnist dataset
XTrain, yTrain, XTest, yTest, numClasses = load.loadDataset("mnist")

#create model from textfile
model = cMod.createModel(filename='myM')

#set number of epochs
epochs = 10

#print summary
print(model.summary())

#fit model and print results
model.fit(XTrain, yTrain, validation_data = (XTest, yTest), epochs = epochs, batch_size = 200)
score = model.evaluate(XTest, yTest, verbose = 0)
print("Accuracy: %.2f%%" % (score[1] * 100))
