# Holographic word representations

import pickle
import numpy as np
from numpy.random import normal, seed
from numpy.fft import fft, ifft
import numpy.linalg as la

seed(42)

length = 2048.0

home = '/Users/dgclark/Documents/dcfiles/manuscripts/new_easton/ngrams/'

fh = open(home+'unigrams.pkl','r')
proto_uni = pickle.load(fh)
fh.close()

fh = open(home+'targetWords.pkl','r')
targs = pickle.load(fh)
fh.close()

fh = open(home+'stoplist_JM.txt','r')
stops = [ word.strip() for word in fh.readlines() ]
fh.close()

theta = sorted(proto_uni.values())[-90000]
uni = dict([ (k,proto_uni[k]) for k in proto_uni if proto_uni[k] >= theta or k in targs ])

def disallowed(w):
    if any([letter in w for letter in "!@#$%^&*(){}1234567890=+[]?."]):
        return True
    if not any([letter in w for letter in [ chr(y) for y in range(65,91)+range(97,123) ]]):
        return True
    else: return False

sorted_rep = sorted([ (uni[k],k) for k in uni if not disallowed(k) and k not in stops ])
print sorted_rep[-10000:]

def leftArg(a):
    return np.roll(a,-1)

def rightArg(a):
    return np.roll(a,1)

def convolve(left,right):
    return ifft(np.multiply(fft(leftArg(left)),fft(rightArg(right))))

def cosim(v1,v2):
    n1 = la.norm(v1)
    n2 = la.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    else:
        return np.dot(v1/n1,v2/n2)

mem = dict([ (k,np.zeros(2048)) for k in targs ])
env = dict([ (k,normal(0,1.0/np.sqrt(length),length)) for k in uni ])

fh = open('allCounts.pkl','r')
con = pickle.load(fh)
fh.close()

for target in targs:
    for context in con[target].keys():
        if env.has_key(context) and not disallowed(context) and context not in stops:
            try:
                mem[target] += (env[context] * con[target][context])
            except KeyError:
                print "Target:", target
                print "Context:", context

# Top 10 neighbors for some words
def top10(w):
    neighbors = {}
    for word in mem:
        if word != w:
            neighbors[word] = cosim(mem[w],mem[word])
    theta = sorted(neighbors.values())[-10]
    return sorted([ (neighbors[word],word) for word in neighbors if neighbors[word] >= theta ])

for word in [ 'dog', 'jacket', 'cake', 'underwear', 'mouse', 'onion', 'flag', 'mango', 'snake', 'arrive' ]:
    print word.upper()
    t10 = top10(word)
    for (co,w) in t10:
        print w, ":", co

fh = open(home+'environment_vectors.pkl','w')
pickle.dump(env,fh)
fh.close()

fh = open(home+'lexical_vectors.pkl','w')
pickle.dump(mem,fh)
fh.close()





