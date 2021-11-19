import sys
from Stack import Stack
from Word import Word
from WordBuffer import WordBuffer

class Config:
        def __init__(self, filename, mcd, dicos):
                self.wb = WordBuffer(filename, mcd)
                self.st = Stack()

        def isFinal(self):
                if self.getStack().getLength() == 1 and self.getStack().top() == 0 and self.getBuffer().getCurrentIndex() >=  self.getBuffer().getLength():  
                        return True
                return False
                
        def getStack(self):
                return self.st
        
        def getBuffer(self):
                return self.wb
    
        def fwd(self):
                if self.getBuffer().endReached() :
                   return False
                self.getBuffer().currentIndex += 1
                return True

        def shift(self):
                if self.getBuffer().endReached() :
                        sys.stderr.write("cannot shift : end of buffer reached\n")
                        return False
                self.getStack().push(self.getBuffer().currentIndex);
                self.fwd()
                return True
    
        def red(self):
                if(self.getStack().isEmpty()):
                        sys.stderr.write("cannot reduce an empty stack !\n")
                        return False
        
                if int(self.getBuffer().getWord(self.getStack().top()).getFeat('GOV')) == Word.invalidGov() :
                        sys.stderr.write("cannot reduce the stack if top element does not have a governor !\n")
                        return False
        
                self.getStack().pop()
                return True
    
        def right(self, label):
                if(self.getStack().isEmpty()):
                        print("cannot make a right move, the stack is empty!")
                        return False
                
                govIndex = self.getStack().top()
                depIndex = self.getBuffer().currentIndex
                self.getBuffer().getCurrentWord().setFeat('LABEL', label)
                self.getBuffer().getCurrentWord().setFeat('GOV', str(govIndex - depIndex))
                self.getBuffer().getWord(self.getStack().top()).addRightDaughter(depIndex)
                self.getStack().push(self.getBuffer().currentIndex)
                res = self.fwd()
                return res

        def left(self, label):
                if(self.getStack().isEmpty()):
                        print("cannot make a left move, the stack is empty!")
                        return False
		
                govIndex = self.getBuffer().currentIndex
                depIndex = self.getStack().top()
                self.getBuffer().getWord(self.getStack().top()).setFeat('LABEL', label)
                self.getBuffer().getWord(self.getStack().top()).setFeat('GOV', str(govIndex - depIndex))
                self.getBuffer().getCurrentWord().addLeftDaughter(depIndex)
                self.getStack().pop()
                return True
    
        def applyMvt(self, mvt):
                mvt_type = mvt[0]
                mvt_label = mvt[1]
                if(mvt_type == 'RIGHT'):
                        return self.right(mvt_label)
                elif(mvt_type == 'LEFT'):
                        return self.left(mvt_label)
                elif(mvt_type == 'SHIFT'):
                        return self.shift()
                elif(mvt_type == 'REDUCE'):
                        return self.red()
                return False

        def getWordWithRelativeIndex(self, container, index):
                if container == 'S' :
                        if index >= self.getStack().getLength() :
                                return None
                        indexInBuffer = self.getStack().array[self.getStack().getLength() - index - 1]
                        return self.getBuffer().getWord(indexInBuffer)
                elif container == 'B' :
                        absoluteIndex = self.getBuffer().getCurrentIndex() + index
                        if absoluteIndex < self.getBuffer().getLength() and absoluteIndex >= 0 :
                                return self.getBuffer().getWord(absoluteIndex)
                        else :
                                return None
                return None
        
        def getFeat(self, featTuple):
                featType = featTuple[0]
                if(featType == 'W'):
                        return self.getWordFeat(featTuple)
                elif(featType == 'C'):
                        return self.getConfFeat(featTuple)

        def getConfFeat(self, featTuple):
                featSubType = featTuple[1]
                if featSubType == 'DIST':
                        return self.getDistFeat(featTuple)
                elif featSubType == 'NLDEP':
                        return self.getNldepFeat(featTuple)
                elif featSubType == 'NRDEP':
                        return self.getNrdepFeat(featTuple)
                elif featSubType == 'LLDEP':
                        return self.getLldepFeat(featTuple)
                elif featSubType == 'LRDEP':
                        return self.getLrdepFeat(featTuple)
                elif featSubType == 'SH':
                        return self.getStackHeightFeat(featTuple)
                return 'NULL'

        def getNlDepFeat(self, featTuple):
                container = featTuple[2]
                index = featTuple[3]

                word = self.getWordWithRelativeIndex(containe, index)
                if word == None :
                        return 'NULL'
                return string(len(word.getLeftDaughters()))

        def getNrDepFeat(self, featTuple):
                container = featTuple[2]
                index = featTuple[3]

                word = self.getWordWithRelativeIndex(containe, index)
                if word == None :
                        return 'NULL'
                return string(len(word.getRightDaughters()))


        def getLlDepFeat(self, featTuple):
                return 'NULL'

        def getLrDepFeat(self, featTuple):
                return 'NULL'

        def getStackHeightFeat(self, featTuple):
                string(self.getStack().getLength())

        def getDistFeat(self, featTuple):
                containerWord1 = featTuple[1]
                indexWord1     = featTuple[2]
                containerWord2 = featTuple[3]
                indexWord2     = featTuple[4]
                word1          = self.getWordWithRelativeIndex(containerWord1, indexWord1)
                word2          = self.getWordWithRelativeIndex(containerWord2, indexWord2)
                
                if word1 == None or word2 == None :
                        return 'NULL'
                return word1.getIndex() - word2.getIndex() 

        def getWordFeat(self, featTuple):
                container = featTuple[1]
                index     = featTuple[2]
                tape      = featTuple[3]

                word = self.getWordWithRelativeIndex(container, index)
                if word == None :
                        return 'NULL'
                return word.getFeat(tape)

        def affiche(self):
                currentIndex = self.getBuffer().getCurrentIndex()
                print('BUFFER = ', end = '')
                for i in range(currentIndex - 2, currentIndex + 2):
                        if((i >= 0) and (i < len(self.getBuffer().array))):
                                if(i == currentIndex):
                                        print('[[', i, ':', self.getBuffer().getWord(i).getFeat('POS'), ']] ', end = ' ')
                                else:
                                        print('[', i, ':', self.getBuffer().getWord(i).getFeat('POS'), '] ', end = ' ')

                print('\nSTACK  = [', end = '')
                st = self.getStack()
                for elt in st.array:
                        print(elt, ' ', end = '')
                print(']')
        

        def extractFeatVec(self, FeatModel):
                featVec = []
                i = 0
                for f in FeatModel.getFeatArray():
#                        print(f, '=', self.getWordFeat(f))
                        featVec.append(self.getFeat(f))
#                        featVec.append(self.getWordFeat(f))
                        i += 1
#                print(featVec)
                return featVec
