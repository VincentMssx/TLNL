import sys
from WordBuffer import WordBuffer
from Word import Word

if len(sys.argv) < 2 :
    print('usage:', sys.argv[0], 'conllFile mcdFile')
    exit(1)


conlluFilename = sys.argv[1]

def isDepProj(wordBuffer, depIndex) :
    govIndex = wordBuffer.getWord(depIndex).getFeat('GOV')
#    print("dep Index = ", depIndex, "gov Index =", govIndex)
    if depIndex < govIndex :
        for currentIndex in range(depIndex + 1, govIndex):
            currentGovIndex =  wordBuffer.getWord(currentIndex).getFeat('GOV')
            if currentGovIndex < depIndex or currentGovIndex > govIndex :
#                print("word not projective :", currentIndex)
                return False
    if govIndex < depIndex :
        for currentIndex in range(govIndex + 1, depIndex):
            currentGovIndex =  wordBuffer.getWord(currentIndex).getFeat('GOV')
            if currentGovIndex < govIndex or currentGovIndex > depIndex :
#                print("word not projective :", currentIndex)
                return False
    return True

def isSentProj(wordBuffer) :
    for currentIndex in range(1, wordBuffer.getLength()):
        if not isDepProj(wordBuffer, currentIndex):
            return False
    return True
            
def printConllSentence(wordBuffer):
    for currentIndex in range(1, wordBuffer.getLength()):
        w = wordBuffer.getWord(currentIndex)
        print(w.getFeat('INDEX'), end = '\t')
        print(w.getFeat('FORM'), end = '\t')
        print(w.getFeat('LEMMA'), end = '\t')
        print(w.getFeat('POS'), end = '\t')
        print(w.getFeat('X1'), end = '\t')
        print(w.getFeat('MORPHO'), end = '\t')
        print(w.getFeat('GOV'), end = '\t')
        print(w.getFeat('LABEL'), end = '\t')
        print(w.getFeat('X2'), end = '\t')
        print(w.getFeat('X3'))
    print()



try:
    conlluFile = open(conlluFilename, encoding='utf-8')
except IOError:
    print(conlluFilename, " : ce fichier n'existe pas")
    exit(1)

tokens = []
wordBuffer = WordBuffer()
wordBuffer.initConll()
sentNb = 1
for ligne in conlluFile:
#    print(ligne, end = '')
    if ligne[0] == '\n' :
#        print("sentence ", sentNb, end = '\t')
        sentNb += 1
        if isSentProj(wordBuffer):
#            print("is projective")
            printConllSentence(wordBuffer)
#        else:
#            print("is not projective")
        wordBuffer.initConll()
        next
    elif ligne[0] == '#' :
        #print("commentaire")
        next
    else :
        ligne = ligne.rstrip()
#                                1	Je	il	PRON	_	Number=Sing|Person=1|PronType=Prs	2	nsubj	_	_
        tokens = ligne.split("\t")
        if '-' not in tokens[0]:
            w = Word()
            index = int(tokens[0])
            w.setFeat('INDEX', tokens[0])
            w.setFeat('FORM', tokens[1])
            w.setFeat('LEMMA', tokens[2])
            w.setFeat('POS', tokens[3])
            w.setFeat('X1', tokens[4])
            w.setFeat('MORPHO', tokens[5])
            w.setFeat('GOV', int(tokens[6]))
            w.setFeat('LABEL', tokens[7])
            w.setFeat('X2', tokens[8])
            w.setFeat('X3', tokens[9])
            w.setFeat('EOS', '0')
            wordBuffer.addWord(w)
            
conlluFile.close();






