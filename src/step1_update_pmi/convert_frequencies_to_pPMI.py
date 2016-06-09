# This is a new version of an old program-- this version uses the latest counts of
# target and context words from the Easton participants.

import pickle
from math import log, exp
import numpy as np
from numpy.linalg import norm

def pluralize(w):
    if w[-1] == 's' or w[-1] == 'z' or w[-2:] == 'sh':
        return w + 'es'
    if w[-1] == 'y':
        return w[:-1] + 'ies'
    return w + 's'

print "Loading joint frequencies..."
fh = open('contextVectors.pkl','r')
joints = pickle.load(fh)
fh.close()

print "Combining vectors for inflected targets..."
def find_inflected(ks):
    tups = []
    for k in ks:
        infl = pluralize(k)
        if infl in ks:
            tups.append((k,infl))
    return tups

inflected = find_inflected([k for k in joints.keys() if k])
print inflected

def combine_inflected(hash,tups):
    for (un,inf) in tups:
        for key in hash[inf]:
            if not hash[un].has_key(key):
                hash[un][key] = hash[inf][key]
            else:
                hash[un][key] += hash[inf][key]
    return hash

joints = combine_inflected(joints,inflected)

print "Loading raw frequencies..."
fh = open('correctedLogUnigrams.pkl','r')
raws = pickle.load(fh)
fh.close()
raws = dict([ (k,exp(raws[k])) for k in raws ])
total_raw = float(sum(raws.values()))
raw_prob = dict([ (key,raws[key]/total_raw) for key in raws ])
raws = None

print "Calculating probability of context words..."
context_prob = {}
for target in joints:
    sum_contexts = float(sum(joints[target].values()))
    context_prob[target] = {}
    for con in joints[target].keys():
        context_prob[target][con] = joints[target][con]/sum_contexts

print "Calculating positive pointwise mutual information for entries in main hash table..."
# Using formula pPMI = max(0,log(P(context|targ)/P(context)))
pPMI = {}
for target in context_prob.keys():
    pPMI[target] = {}
    for con in context_prob[target].keys():
        pConGivenTarg = context_prob[target][con]
        try:
            pCon = raw_prob[con]
        except KeyError:
            pCon = 0
        try:
            ppmi = log(pConGivenTarg) - log(pCon)
        except ValueError:
            ppmi = 0
        pPMI[target][con] = max(0,ppmi)


print "Saving pPMI to file pPMI_all.pickle"
fh = open("pPMI_all.pickle",'w')
pickle.dump(pPMI,fh)
fh.close()

def cosim_hash(h1,h2):
    allKeys = sorted(list(set(h1.keys()).union(set(h2.keys()))))
    v1 = np.zeros(len(allKeys))
    v2 = np.zeros(len(allKeys))
    for (i,k) in enumerate(allKeys):
        if h1.has_key(k):
            v1[i] = h1[k]
        if h2.has_key(k):
            v2[i] = h2[k]
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 != 0 and n2 != 0:
        v1 = v1/n1
        v2 = v2/n2
        return np.dot(v1,v2)
    else:
        return 0.0

def top_items(s):
    coses = [(cosim_hash(pPMI[k],pPMI[s]),k) for k in pPMI.keys() if k != s]
    thetas = sorted([ c for (c,k) in coses ])
    theta = thetas[-15]
    return sorted([ (c,k) for (c,k) in coses if c >= theta ],reverse=True)

for word in [ 'eagle', 'socks', 'jacket', 'asparagus', 'brussels_sprouts', 'spaghetti', 'lion', 'fun', 'arrive', 'straight' ]:
    print word, ':'
    top15 = top_items(word)
    for (co,item) in top15:
        print '\t', item, co

        
