class Stack:
    def __init__(self):
        self.array = []

    def isEmpty(self):
        if(len(self.array) == 0):
            return True
        else:
            return False

    def empty(self):
        self.array = []
        
    def push(self, elt):
        self.array.append(elt)

    def pop(self):
        if(self.isEmpty()):
            print("cannot pop an empty stack");
        else:
            return(self.array.pop())

    def top(self):
        if(self.isEmpty() == False):
            return self.array[len(self.array) - 1]

    def getLength(self):
        return len(self.array)

    def affiche(self):
        print("---- bottom ----")
        for elt in self.array:
            print(elt)
        print("----   top  ----")

    
            
    
