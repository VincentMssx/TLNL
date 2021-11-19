import sys
from Dicos import Dicos
from Mcd import Mcd


if len(sys.argv) < 4 :
    print('usage:', sys.argv[0], 'mcf_file mcd_file dico_file')
    exit(1)

mcfFileName = sys.argv[1]
mcdFileName = sys.argv[2]
dicoFileName = sys.argv[3]

mcd = Mcd(mcdFileName)

print('populating dicos from file ', mcfFileName)
dicos = Dicos(mcd)
dicos.populateFromMcfFile(mcfFileName, mcd, verbose=False)
print('saving dicos in file ', dicoFileName)
dicos.printToFile(dicoFileName)


