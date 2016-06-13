class VF_list():    def __init__(self,filename):        """ Create a VF list object """        with open(filename,'r') as fh:            lines = [ line.strip() for line in fh.readlines()]            self.lines = [ line for line in lines if line ]    def __call__(self):        """ Return all the items from a VF list, including comments, intrusions, etc. """        return self.lines    def valid(self):        """ Return all the items from a VF list if the first character is alphabetical """        return [ el for el in self.lines if el[0].isalpha()]    def raw(self):        """ Return the raw score on a VF list """        return len(self.valid())    def reps(self):        """ Return the count of repeated items (perseverations) """        return sum([ el == '#' for el in self.lines])    def intrusions(self):        """ Return the count of intrusions """        return sum([ el == '!' for el in self.lines ])    def repair_gaps(self):        """ Find spaces and replace them with _ """        temp = []        for item in self.lines:            if item[0] in '#!@-%&?':                bitz = item.split()                woid = '_'.join(bitz[1:])                temp.append(' '.join([bitz[0],woid]))            elif item[0] == '(':                temp.append(item)            elif ' ' in item:                temp.append('_'.join(item.split()))            else:                temp.append(item)        self.lines = temp    def items(self):        '''Get all of the items (non-comment lines) from the list'''        self.repair_gaps()        tems = []        for i in self.lines:            if i[0] in '#!@%&?':                tems.append(i[1:].strip())            elif i[0] not in '(-':                tems.append(i.strip())        return tems