from Dico import Dico

class Dicos:
        def __init__(self, mcd=False, fileName=False):
                self.content = {}
                if mcd :
                        self.initializeWithMcd(mcd)
                if fileName :
                        self.initializeWithDicoFile(fileName)

        def initializeWithMcd(self, mcd):
                for index in range(mcd.getNbCol()):
                        if(mcd.getColStatus(index) == 'KEEP') and (mcd.getColType(index) == 'SYM') :
                                dico = self.addDico(mcd.getColName(index))
                                dico.add('NULL')
                                dico.add('ROOT')
                                
        def initializeWithDicoFile(self, fileName):
                try:
                        dicoFile = open(fileName, encoding='utf-8')
                except IOError:
                        print(fileName, 'does not exist')
                        exit(1)
                for ligne in dicoFile:
                        if ligne[0] == '#' and ligne[1] == '#' :
                                currentDicoName = ligne[2:-1]
                                #                                        currentDico = self.getDico(currentDicoName)
                                currentDico = self.addDico(currentDicoName)
                        else:
                                symbol = ligne[:-1]
                                currentDico.add(symbol)
                dicoFile.close()
                

        def populateFromMcfFile(self, mcfFilename, mcd, verbose=False):
                try:
                        mcfFile = open(mcfFilename, encoding='utf-8')
                except IOError:
                        print('cannot open', mcfFilename)
                        exit(1)
                        tokens = []
                for ligne in mcfFile:
                        ligne = ligne.rstrip('\n\r')
                        tokens = ligne.split("\t")
                        for i in range(0, len(tokens)):
                                if mcd.getColType(i) == 'SYM' and mcd.getColStatus(i) == 'KEEP':
                                        self.add(mcd.getColName(i), tokens[i])
                mcfFile.close();
                for e in self.content:
                        print('DICO', e, ':\t', self.content[e].getSize(), 'entries')
                                                        
#        def populateFromConlluFile(self, conlluFilename, verbose=False):
#                try:
#                        conlluFile = open(conlluFilename, encoding='utf-8')
#                except IOError:
#                        print('cannot open', conlluFilename)
#                        exit(1)
#                mcd = (('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
#                tokens = []
#                for ligne in conlluFile:
#                        if ligne[0] != '\n' and ligne[0] != '#' :
#                                tokens = ligne.split("\t")
#                                for i in range(0, len(tokens)):
#                                        if mcd[i][1] == 'SYM' :
#                                                if not tokens[i] in self.content[mcd[i][0]] :
#                                                        self.content[mcd[i][0]].append(tokens[i])
#                                                        if(verbose): print('in module:', __name__, 'adding value ', tokens[i], 'to dico', mcd[i][0]) 
#                conlluFile.close();
#                for e in self.content:
#                        print('DICO', e, ':\t', len(self.content[e]), 'entries')
                                                        
        def print(self):
                for dicoName in self.content.keys():
                        self.content[dicoName].print()

        def printToFile(self, filename):
            try:
                dicoFile = open(filename, 'w', encoding='utf-8')
            except IOError:
                print('cannot open', filename)
                exit(1)
            for dicoName in self.content.keys():
                    self.content[dicoName].printToFile(dicoFile)
            dicoFile.close()

        def getDico(self, dicoName):
                if not dicoName in self.content :
                        return None
                return self.content[dicoName]

        def addDico(self, dicoName):
                if dicoName in self.content :
                        return self.content[dicoName]
                dico = Dico(dicoName)
                self.content[dicoName] = dico 
                return dico

        def getCode(self, dicoName, symbol) :
                dico = self.getDico(dicoName)
                if dico == None :
                        return None
                return dico.getCode(symbol)

        def getSymbol(self, dicoName, code) :
                dico = self.getDico()
                if dico == None :
                        return None
                return dico.getSymbol()

        def add(self, dicoName, symbol) :
                dico = self.getDico(dicoName)
                if dico == None :
                        return None
                return dico.add(symbol)

