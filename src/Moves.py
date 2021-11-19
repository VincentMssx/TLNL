import numpy as np

class Moves:
    def __init__(self, dicos):
        self.dicoLabels = dicos.getDico('LABEL')
        if not self.dicoLabels :
            print("cannot find LABEL in dicos")
            exit(1)
        self.nb = 2 * self.dicoLabels.getSize() + 3

    def getNb(self):
        return self.nb

    def mvtCode(self, mvt):
        mvtType = mvt[0]
        mvtLabel = mvt[1]
        if(mvtType == 'SHIFT'):  return 0
        if(mvtType == 'REDUCE'): return 1
        if(mvtType == 'ROOT'):   return 2
        labelCode = self.dicoLabels.getCode(mvtLabel)
        if not labelCode :
            labelCode = self.dicoLabels.getCode('NULL')
            #print("cannot compute code of movement ", mvt, "label ", mvtLabel, "unknown")
            #exit(1)
        if(mvtType == 'RIGHT'):  return 3 + 2 * labelCode
        if(mvtType == 'LEFT'):   return 3 + 2 * labelCode + 1

    def mvtDecode(self, mvt_Code):
        if(mvt_Code == 0) : return ('SHIFT', 'NULL')
        if(mvt_Code == 1) : return ('REDUCE', 'NULL')
        if(mvt_Code == 2) : return ('ROOT', 'NULL')
        if mvt_Code % 2 == 0 : #LEFT
            labelCode = int((mvt_Code - 4) / 2)
            return ('LEFT', self.dicoLabels.getSymbol(labelCode))
        else :
            labelCode = int((mvt_Code - 3)/ 2)
            return ('RIGHT', self.dicoLabels.getSymbol(labelCode))

    def buildOutputVector(self, mvt):
        outputVector = np.zeros(self.nb, dtype="int32")
        codeMvt = self.mvtCode(mvt)
        outputVector[codeMvt] = 1
        return outputVector
