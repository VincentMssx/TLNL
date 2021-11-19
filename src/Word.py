class Word:
    def __init__(self):
        self.featDic = {}         # dictionnaire dans lequel sont stockés les word features
        self.leftDaughters = []   # liste des indices des dépendants gauches
        self.rightDaughters = []  # liste des indices des dépendants droits
        self.index = self.invalidIndex()

    def getFeat(self, featName):
        if(not featName in self.featDic):
            print('WARNING : feat', featName, 'does not exist')
            return None
        else:
            return self.featDic[featName]

    def setFeat(self, featName, featValue):
        self.featDic[featName] = featValue

    def addLeftDaughter(self, index):
        self.leftDaughters.append(index)

    def addRightDaughter(self, index):
        self.rightDaughters.append(index)

    def getIndex(self) :
        return self.index

    def getRightDaughters(self):
        return self.rightDaughters
    
    def getLeftDaughters(self):
        return self.leftDaughters
    
    
    def affiche(self, mcd):
        first = True
        for columnNb in range(mcd.getNbCol()):
            if mcd.getColStatus(columnNb) == 'KEEP':
                if first:
                    first = False
                else:
                    print('\t', end='')
                print(self.getFeat(mcd.getColName(columnNb)), end='')
#        print('')

    @staticmethod
    def fakeWordConll():
        w = Word()
        return w

        
    @staticmethod
    def fakeWord(mcd):
        w =Word()
        for elt in mcd.getArray():
            (col, feat, type, status) = elt
            w.setFeat(feat, 'ROOT')
        w.setFeat('GOV', '0')
        return w

    @staticmethod
    def invalidIndex():
        return 123456789

    @staticmethod
    def invalidGov():
        return 123456789

    @staticmethod
    def invalidLabel():
        return ''

