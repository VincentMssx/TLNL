import sys
from Mcd import Mcd
from WordBuffer import WordBuffer
from Word import Word

if len(sys.argv) < 6 :
    print('usage:', sys.argv[0], 'ref_mcf hyp_mcf ref_mcd hyp_mcd lang')
    exit(1)

refFileName = sys.argv[1]
hypFileName = sys.argv[2]
refMcdFileName = sys.argv[3]
hypMcdFileName = sys.argv[4]
lang = sys.argv[5]

if len(sys.argv) == 7:
    verbose = True
else:
    verbose = False


#print('reading mcd from file :', refMcdFileName)
refMcd = Mcd(refMcdFileName)

#print('reading mcd from file :', hypMcdFileName)
hypMcd = Mcd(hypMcdFileName)

GovColIndex = refMcd.locateCol('GOV')
if(GovColIndex == None):
    print("cannot locate column GOV in mcd :", refMcdFileName)

LabelColIndex = refMcd.locateCol('LABEL')
if(LabelColIndex == None):
    print("cannot locate column LABEL in mcd :", refMcdFileName)

GovColIndex = hypMcd.locateCol('GOV')
if(GovColIndex == None):
    print("cannot locate column GOV in mcd :", hypMcdFileName)

LabelColIndex = hypMcd.locateCol('LABEL')
if(LabelColIndex == None):
    print("cannot locate column LABEL in mcd :", hypMcdFileName)

refWordBuffer = WordBuffer(refFileName, refMcd)
refWordBuffer.readAllMcfFile()

hypWordBuffer = WordBuffer(hypFileName, hypMcd)
hypWordBuffer.readAllMcfFile()

govCorrect = 0
labelCorrect = 0

hypTotal = {}
refTotal = {}
refInterHypTotal = {}

hypSize = hypWordBuffer.getLength()
for index in range(hypSize):
    refWord = refWordBuffer.getWord(index)
    hypWord = hypWordBuffer.getWord(index)
    refGov = refWord.getFeat("GOV")
    hypGov = hypWord.getFeat("GOV")
    refLabel = refWord.getFeat("LABEL")
    hypLabel = hypWord.getFeat("LABEL")
    
    if hypLabel in hypTotal :
        hypTotal[hypLabel] += 1
    else:
        hypTotal[hypLabel] = 1
        
    if refLabel in refTotal :
        refTotal[refLabel] += 1
    else:
        refTotal[refLabel] = 1
    if refGov == hypGov :
        govCorrect += 1
        if refLabel == hypLabel :
            labelCorrect += 1
            if refLabel in refInterHypTotal :
                refInterHypTotal[refLabel] += 1
            else:
                refInterHypTotal[refLabel] = 1

LAS = 100 * labelCorrect / hypSize
UAS = 100 * govCorrect / hypSize

print("%s\t%.2f\t%.2f" % (lang, LAS, UAS))


if verbose :
    print("------------------------------")
    print("label\tprec\trec\tfscore")
    print("------------------------------")
    for label in refInterHypTotal:
        precision = refInterHypTotal[label] / hypTotal[label]
        recall = refInterHypTotal[label] / refTotal[label]
        fscore = 2 * precision * recall / (precision + recall)
        print("%s\t%.2f\t%.2f\t%.2f" % (label, precision, recall, fscore) )


#    print("REF GOV = ", refGov, "HYP GOV = ", hypGov, "REF LABEL = ", refLabel, "HYP LABEL = ", hypLabel)
