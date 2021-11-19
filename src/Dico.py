class Dico:
        def __init__(self, name):
                self.name = name
                self.hash = {}
                self.array = []

        def add(self, symbol):
                if symbol in self.hash :
                        return self.hash[symbol]
                self.hash[symbol] = len(self.array)
                self.array.append(symbol)
                return len(self.array) - 1

        def getCode(self, symbol):
                if not symbol in self.hash :
                        return None
                return self.hash[symbol]

        def getSymbol(self, code):
                if code >= len(self.array) :
                        return None
                return self.array[code]

        def getSize(self):
                return len(self.array)

        def printToFile(self, dicoFile):
                dicoFile.write('##')
                dicoFile.write(self.name)
                dicoFile.write('\n')
                for i in range(len(self.array)):
                    dicoFile.write(self.array[i])
                    dicoFile.write('\n')

        
        def print(self):
                print('##', self.name, sep = '')
                for i in range(len(self.array)):
                        print(self.array[i], i)
