import sys
import Oracle
from Dicos import Dicos
from Config import Config
from Word import Word
from Mcd import Mcd
from Moves import Moves
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from FeatModel import FeatModel
import numpy as np



def prepareWordBufferForDecode(buffer):
    """Add to every word of the buffer features GOVREF and LABELREF.

    GOVEREF is a copy of feature GOV and LABELREF a copy of LABEL
    GOV and LABEL are set to initialization values
    """
    for word in buffer.array:
        word.setFeat('GOV', str(Word.invalidGov()))
        word.setFeat('LABEL', Word.invalidLabel())


verbose = False
if len(sys.argv) != 7 :
    print('usage:', sys.argv[0], 'mcf_file model_file dicos_file feat_model mcd_file words_limit')
    exit(1)

mcf_file =       sys.argv[1]
model_file =     sys.argv[2]
dicos_file =     sys.argv[3]
feat_model =     sys.argv[4]
mcd_file =       sys.argv[5]
wordsLimit = int(sys.argv[6])

    
sys.stderr.write('reading mcd from file :')
sys.stderr.write(mcd_file)
sys.stderr.write('\n')
mcd = Mcd(mcd_file)

sys.stderr.write('loading dicos\n')
dicos = Dicos(fileName=dicos_file)

moves = Moves(dicos)

sys.stderr.write('reading feature model from file :')
sys.stderr.write(feat_model)
sys.stderr.write('\n')
featModel = FeatModel(feat_model, dicos)

sys.stderr.write('loading model :')
sys.stderr.write(model_file)
sys.stderr.write('\n')
model = load_model(model_file)

inputSize = featModel.getInputSize()
outputSize = moves.getNb()

c = Config(mcf_file, mcd, dicos)
numSent = 0
verbose = False
numWords = 0

while c.getBuffer().readNextSentence()  and numWords < wordsLimit :
    c.getStack().empty()
    prepareWordBufferForDecode(c.getBuffer())
    numWords += c.getBuffer().getLength()

    while True :
        featVec = c.extractFeatVec(featModel)
        inputVector = featModel.buildInputVector(featVec, dicos)
        outputVector = model.predict(inputVector.reshape((1,inputSize)), batch_size=1, verbose=0, steps=None)
        mvt_Code = outputVector.argmax()
        mvt = moves.mvtDecode(mvt_Code)

        if(verbose == True) :
            print("------------------------------------------")
            c.affiche()
            print('predicted move', mvt[0], mvt[1])
            print(mvt, featVec)

        res = c.applyMvt(mvt)
        if not res :
            sys.stderr.write("cannot apply predicted movement\n")
            mvt_type = mvt[0]
            mvt_label = mvt[1]
            if mvt_type != "SHIFT" :
                sys.stderr.write("try to force SHIFT\n")
                res = c.shift()
                if res == False :
                    sys.stderr.write("try to force REDUCE\n")
                    res = c.red()
                    if res == False :
                        sys.stderr.write("abort sentence\n")
                        break
        if(c.isFinal()):
            break
    for i in range(1, c.getBuffer().getLength()):
        w = c.getBuffer().getWord(i)
        w.affiche(mcd)
        print('')
#        print('\t', w.getFeat("GOV"), end='\t')
#        print(w.getFeat("LABEL"))

    numSent += 1
#    if numSent % 10 == 0:
#        print ("Sent : ", numSent)






