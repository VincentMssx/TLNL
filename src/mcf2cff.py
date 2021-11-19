import sys
import Oracle
from Moves import Moves
from Mcd import Mcd
from FeatModel import FeatModel
from Dicos import Dicos
from Config import Config
from Word import Word
import numpy as np

def prepareWordBufferForTrain(buffer):
    """Add to every word of the buffer features GOVREF and LABELREF.

    GOVEREF is a copy of feature GOV and LABELREF a copy of LABEL
    GOV and LABEL are set to initialization values
    """
    for word in buffer.array:
        word.setFeat('GOVREF', word.getFeat('GOV'))
        word.setFeat('GOV', str(Word.invalidGov()))
        word.setFeat('LABELREF', word.getFeat('LABEL'))
        word.setFeat('LABEL', Word.invalidLabel())

def prepareData(mcd, mcfFile, featModel, moves, filename, wordsLimit) :
    try:
        dataFile = open(filename, 'w', encoding='utf-8')
    except IOError:
        print('cannot open', filename)
        exit(1)

    dataFile.write(str(inputSize))
    dataFile.write("\n")
    dataFile.write(str(outputSize))
    dataFile.write("\n")
    c = Config(mcfFile, mcd, dicos)
    numSent = 0
    numWords = 0
    while c.getBuffer().readNextSentence() and numWords < wordsLimit:
        numWords += c.getBuffer().getLength()
        numSent += 1
#        print(">>>>>>>>>>>>> Sent", numSent, " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        prepareWordBufferForTrain(c.getBuffer())
        while True :
            mvt = Oracle.oracle(c)
            outputVector = moves.buildOutputVector(mvt)
            featVec = c.extractFeatVec(featModel)
            inputVector = featModel.buildInputVector(featVec, dicos)
            np.savetxt(dataFile, inputVector, fmt="%s", delimiter='  ', newline=' ')
            dataFile.write('\n')
            np.savetxt(dataFile, outputVector, fmt="%s", delimiter='  ', newline=' ')
            dataFile.write('\n')

            if(verbose == True) :
                print("------------------------------------------")
                c.affiche()
                print('oracle says', mvt[0], mvt[1])
                print(mvt, featVec)

            # c.getBuffer().affiche(mcd)
            res = c.applyMvt(mvt)
            if(res == False): print("cannot apply movement")
            if(c.isFinal()):
#                print("is final is true")
                break


            
if len(sys.argv) < 5 :
    print('usage:', sys.argv[0], 'mcf_file feat_model_file mcd_file dicos_file data_file words_limit')
    exit(1)

mcfFileName =       sys.argv[1]
featModelFileName = sys.argv[2]
mcdFileName =       sys.argv[3]
dicosFileName =     sys.argv[4]
dataFileName =      sys.argv[5]
wordsLimit =    int(sys.argv[6])
verbose = False

print('reading mcd from file :', mcdFileName)
mcd = Mcd(mcdFileName)

print('reading dicos from file :', dicosFileName)
dicos = Dicos(fileName = dicosFileName)

#dicos.populateFromMcfFile(mcfFileName, mcd, verbose=False)
#print('saving dicos in file :', dicosFileName)
#dicos.printToFile(dicosFileName)

moves = Moves(dicos)

print('reading feature model from file :', featModelFileName)
featModel = FeatModel(featModelFileName, dicos)

inputSize = featModel.getInputSize()
outputSize = moves.getNb()
print('input size = ', inputSize, 'outputSize =' , outputSize)

print('preparing training data')
prepareData(mcd, mcfFileName, featModel, moves, dataFileName, wordsLimit)



