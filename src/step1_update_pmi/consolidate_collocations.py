import pickle
import os
import sys


# dir = '/Users/dgclark/Documents/dcfiles/manuscripts/new_easton/ngrams/'
dir = '/ifshome/dclark/ngrams/'
begin = 'collocations'

if len(sys.argv) > 1:
    begin = begin + '_' + sys.argv[1]
files = sorted([  f for f in os.listdir(dir) if f.startswith(begin) and f.endswith('.pkl') ],reverse=True)

def consolidate_hashes(*args):
    mokin = {}
    for h in args:
        for target in h.keys():
            if not mokin.has_key(target):
                mokin[target] = {}
            try:
                for con in h[target].keys():
                    try:
                        mokin[target][con] += h[target][con]
                    except KeyError:
                        mokin[target][con] = h[target][con]
            except:
                for side in range(2):
                    for con in h[target][side].keys():
                        try:
                            mokin[target][con] += h[target][side][con]
                        except KeyError:
                            mokin[target][con] = h[target][side][con]
    return mokin

allContexts = []
allCounts = { 'dog': ({},{}) }
for file in files:
    print file
    fh = open(file,'r')
    kurent = pickle.load(fh)
    fh.close()
    ks = kurent.keys()
    # print ks[0], kurent[ks[0]]
    allCounts = consolidate_hashes(allCounts,kurent)
    for k in allCounts.keys():
        allContexts.extend(allCounts[k].keys())

print allCounts['dog']

if len(sys.argv)< 2:
    allContexts = sorted(list(set(allContexts)))
    fh = open('allContextWords.pkl','w')
    pickle.dump(allContexts,fh)
    fh.close()
    vectorFile = 'contextVectors.pkl'
else:
    vectorFile = 'collocations_all_'+sys.argv[1]+'.pkl'

fh = open(vectorFile,'w')
pickle.dump(allCounts,fh)
fh.close()



            

