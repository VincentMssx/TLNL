import sys
from WordBuffer import WordBuffer
from Word import Word
from Mcd import Mcd

if len(sys.argv) < 3 :
    print('usage:', sys.argv[0], 'conllFile mcdFile')
    exit(1)

def simplifyLabel(label):
    simpleLabel = []
    for i in range(len(label)):
        if not label[i] == ':' :
            simpleLabel.append(label[i])
        else :
            break
    return ''.join(simpleLabel)
            

conlluFilename = sys.argv[1]
mcdFilename = sys.argv[2]

mcd = Mcd(mcdFilename)

try:
    conlluFile = open(conlluFilename, encoding='utf-8')
except IOError:
    print(conlluFilename, " : ce fichier n'existe pas")
    exit(1)

tokens = []
wordBuffer = WordBuffer()
for ligne in conlluFile:
    if ligne[0] == '\n' :
        wordBuffer.getWord(wordBuffer.currentIndex - 1).setFeat('EOS', '1')
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
            w.setFeat('GOV', int(tokens[6]) - index)
            label = simplifyLabel(tokens[7])
            w.setFeat('LABEL', label)
            w.setFeat('X2', tokens[8])
            w.setFeat('X3', tokens[9])
            w.setFeat('EOS', '0')
            wordBuffer.addWord(w)

conlluFile.close();

wordBuffer.affiche(mcd)





