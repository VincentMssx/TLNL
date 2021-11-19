import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np

def readData(dataFilename) :
    allX = []
    allY = []
    try:
#        dataFile = open(dataFilename, encoding='utf-8')
        dataFile = open(dataFilename)
    except IOError:
        print(dataFilename, " : ce fichier n'existe pas")
        exit(1)


        
    inputSize = int(dataFile.readline())
    print("input size = ", inputSize)
    outputSize = int(dataFile.readline())
    print("output size = ", outputSize)

    inputLine = True
    for ligne in dataFile:
#        print(ligne)
        vector = ligne.split()
        vector[:] = list(map(int, vector))
        if inputLine == True:
            #print("input ", vector)
            allX.append(vector)
            inputLine = False
        else:
            #print("output ", vector)
            allY.append(vector)
            inputLine = True
    # x_train and y_train are Numpy arrays 
    x_train = np.array(allX)
    y_train = np.array(allY)
    return (inputSize, outputSize, x_train, y_train)



if len(sys.argv) < 3 :
    print('usage:', sys.argv[0], 'cffTrainFileName cffDevFileName kerasModelFileName')
    exit(1)

cffTrainFileName =   sys.argv[1]
cffDevFileName =     sys.argv[2]
kerasModelFileName = sys.argv[3]

inputSize, outputSize, x_train, y_train = readData(cffTrainFileName)
devInputSize, devOutputSize, x_dev, y_dev = readData(cffDevFileName)


model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=inputSize))
model.add(Dropout(0.4))
model.add(Dense(units=outputSize, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_dev,y_dev))


#if len(sys.argv) == 5 :
#    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_dev,y_dev))
#else :
#    model.fit(x_train, y_train, epochs=10, batch_size=32)

model.save(kerasModelFileName)
        

