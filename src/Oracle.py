from Config import Config
from Dicos import Dicos
from Word import Word

def check_all_dependents_of_word_in_ref_are_in_hyp(c, wordIndex):
    """As its name suggests, this function checks that all the dependents of a word have been found.

    this function is called by the oracle to predict a ROOT and a REDUCE action
    """
    depIndex = wordIndex - 1
#    print('target =', wordIndex)
    # look for a dependent of word to its left in reference
    while (depIndex >=0) :
#        print('depIndex = ', depIndex)
        govRefIndex = int(c.getBuffer().getWord(depIndex).getFeat('GOVREF')) + depIndex
#        print("govRefIndex = ", govRefIndex)
        if govRefIndex == wordIndex : # dep is a dependent of word in ref 
            #check that dep has the same governor in hyp 
            govHypIndex = int(c.getBuffer().getWord(depIndex).getFeat('GOV')) + depIndex
#            print(depIndex, 'is dependent ')
            if govHypIndex != govRefIndex :
#                print('wrong gov (', govHypIndex, ')');
                return False
        depIndex -= 1

    sentenceChange = False
    depIndex = wordIndex + 1
    while depIndex < c.getBuffer().getLength() :
#        print('depIndex = ', depIndex)
        govRefIndex = int(c.getBuffer().getWord(depIndex).getFeat('GOVREF')) + depIndex
#        print("govRefIndex = ", govRefIndex)
        if(govRefIndex == wordIndex): # dep is a dependent of word in ref
            govHypIndex = int(c.getBuffer().getWord(depIndex).getFeat('GOV')) + depIndex
#            print(depIndex, 'is dependent ')
            if govHypIndex != govRefIndex :
#                print('wrong gov (', govHypIndex, ')');
                return False
        depIndex += 1

    return True

def oracle(c):
    if(c.getStack().isEmpty()):
        return ('SHIFT', '')
    
    s0_index = c.getStack().top()
    b0_index = c.getBuffer().getCurrentIndex()
#    print("s0_index = ", s0_index)
    s0_gov_index = int(c.getBuffer().getWord(s0_index).getFeat('GOVREF')) + s0_index
    s0_label = c.getBuffer().getWord(s0_index).getFeat('LABELREF')
#    print('s0_index = ', s0_index, 'b0_index = ', b0_index, 's0_gov_index = ', s0_gov_index, 'b0_gov_index = ', b0_gov_index, 's0 label =', s0_label)

    if(s0_gov_index == b0_index):
        return ('LEFT', c.getBuffer().getWord(s0_index).getFeat('LABELREF'))

    if(b0_index < c.getBuffer().getLength()):
        b0_gov_index = int(c.getBuffer().getWord(b0_index).getFeat('GOVREF')) + b0_index
        if(b0_gov_index == s0_index):
            return ('RIGHT', c.getBuffer().getWord(b0_index).getFeat('LABELREF'))

    if((c.getStack().getLength() > 1) and
       check_all_dependents_of_word_in_ref_are_in_hyp(c, s0_index) and # word on top must have all its dependents
       (int(c.getBuffer().getWord(c.getStack().top()).getFeat('GOV')) != Word.invalidGov())):                   # word on top of the stack has a governor 
        return('REDUCE', '')

    #print("no movement possible return SHIFT")
    if not c.getBuffer().endReached(): 
        return('SHIFT', '')
    print("The machine is stucked")
    exit(1)
    


