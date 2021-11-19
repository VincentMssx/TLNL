class Mcd:
    def __init__(self, mcdFilename):
        self.array = self.readMcdFile(mcdFilename)

    def readMcdFile(self, mcdFilename):
        try:
            mcdFile = open(mcdFilename, encoding='utf-8')
        except IOError:
            print(mcdFilename, " : ce fichier n'existe pas")
            exit(1)
        colDescriptionArray = []
        for ligne in mcdFile:
            (col, name, type, status) = ligne.split()
            #print("col = ", col, "name = ", name, "type = ", type, "status =", status)
            if(status != "KEEP" and status != "IGNORE"):
                print("error while reading mcd file : ", mcdFilename, "status :", status, "undefined")
                exit(1)
            if(type != "INT" and type != "SYM"):
                print("error while reading mcd file : ", mcdFilename, "type :", type, "undefined")
                exit(1)
            colDescriptionArray.append((int(col), name, type, status))
        mcdFile.close()
        return colDescriptionArray

    def getNbCol(self):
        return len(self.array)
        
    def getArray(self):
        return self.array

    def getColName(self, colIndex):
        return self.array[colIndex][1]

    def getColType(self, colIndex):
        return self.array[colIndex][2]

    def getColStatus(self, colIndex):
        return self.array[colIndex][3]

    def locateCol(self, name):
        for colIndex in range(self.getNbCol()):
            if self.array[colIndex][1] == name:
                return colIndex
        return None
            
    
