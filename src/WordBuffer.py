from Word import Word

class WordBuffer:
        def __init__(self, mcfFileName=None, mcd=None):
                self.currentIndex = 0
                self.array = []
                self.mcd = mcd
                self.mcfFile = None
                if(mcfFileName):
                        try:
                                self.mcfFile = open(mcfFileName, encoding='utf-8')
                        except IOError:
                                print(mcfFileName, " : ce fichier n'existe pas")
                                exit(1)

        def empty(self):
                self.currentIndex = 0
                self.array = []

        def initConll(self):
                self.empty()
                self.addWord(Word.fakeWordConll())

        def init(self, mcd):
                self.empty()
                self.addWord(Word.fakeWord())
                
        def addWord(self, w):
                self.index = len(self.array)
                self.array.append(w)

        def affiche(self, mcd):
                for w in self.array:
                        w.affiche(mcd)
                        print('')

        def getLength(self):
                return len(self.array)
        
        def getCurrentIndex(self):
                return self.currentIndex

        def getWord(self, index):
                if index >= len(self.array):
                   return None
                return self.array[index]

        def getCurrentWord(self):
                return self.getWord(self.currentIndex)

        def readNextWord(self):
                line = self.mcfFile.readline()
                if line == "" :
                        return None

                line = line.rstrip()
                tokens = line.split("\t")
                w = Word()
                for i in range(0, len(tokens)):
                        w.setFeat(self.mcd.getColName(i), tokens[i])
                self.addWord(w)
                return w
                        
        
        def readNextSentence(self):
                self.currentIndex = 0
                self.array = []
                self.addWord(Word.fakeWord(self.mcd))
                while True:
                        w = self.readNextWord()
                        if w == None :
                                return False
                        if w.getFeat('EOS') == '1':
                                return True
                
        def endReached(self):
                if(self.getCurrentIndex() >= self.getLength()):
                        return True
                else:
                        return False

        def readAllMcfFile(self):
                tokens = []
                for ligne in self.mcfFile:
                        ligne = ligne.rstrip()
                        tokens = ligne.split("\t")
                        w = Word()
                        for i in range(0, len(tokens)):
                                w.setFeat(self.mcd.getColName(i), tokens[i])
                        self.addWord(w)
                self.mcfFile.close();
