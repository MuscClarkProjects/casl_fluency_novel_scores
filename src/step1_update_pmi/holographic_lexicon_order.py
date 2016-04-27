# Holographic word representations

import pickle
import os
import numpy as np
from numpy.random import normal, seed
from numpy.fft import fft, ifft
import numpy.linalg as la
from time import time, sleep
from sys import argv
from numba import jit

ngram = int(argv[1])
csv_no = int(argv[2])

seed(42)

length = 2048.0

# home = '/Users/dgclark/Documents/dcfiles/manuscripts/new_easton/ngrams/'
# tmp = home
home = '/ifshome/dclark/ngrams/'
tmp = '/ifs/tmp/'

print "Loading list of target words"
fh = open(home+'targetWords.pkl','r')
targs = pickle.load(fh)
fh.close()

print "Loading list of stop words"
fh = open(home+'stoplist_JM.txt','r')
stops = [ word.strip() for word in fh.readlines() ]
fh.close()

print "Loading environment vectors... this will take a while"
fh = open(home+'environment_vectors.pkl','r')
env = pickle.load(fh)
fh.close()

if 'phi.pkl' in os.listdir(home):
    print "Loading position marker vector..."
    fh = open(home+'phi.pkl','r')
    phi = pickle.load(fh)
    fh.close()
else:
    print "No position marker vector found, creating one and saving as phi.pkl"
    phi = normal(0,1/np.sqrt(length),length)
    fh = open(home+'phi.pkl','w')
    pickle.dump(phi,fh)
    fh.close()

def disallowed(w):
    if any([letter in w for letter in "!@#$%^&*(){}1234567890=+[]?."]):
        return True
    if not any([letter in w for letter in [ chr(y) for y in range(65,91)+range(97,123) ]]):
        return True
    else: return False

@jit
def leftArg(a):
    return np.roll(a,-1)

@jit
def rightArg(a):
    return np.roll(a,1)

@jit
def convolve(left,right):
    return ifft(np.multiply(fft(leftArg(left)),fft(rightArg(right))))

@jit
def convolve_sequence_aux(el):
    init = convolve(el[0],el[1])
    n = len(el)
    if n > 2:
        for i in range(2,n):
            init = convolve(init,el[i])
    return init

def convolve_sequence(el,i,count):
    """ el is the ngram in list form (along with numerical data), i is the index of the target word,
        count is the number of times the ngram occurred during a given year.
        Returns accumulation of convolutions of all subsequences.
    """
    n = len(el)
    acc = np.zeros(2048)
    try:
        vectors = [ env[word] for word in el ]
    except KeyError:
        # print "Failed to retrieve vector for at least one word in this list:", el
        return acc
    vectors[i] = phi
    acc += (count * convolve_sequence_aux(vectors))
    return acc

def cosim(v1,v2):
    n1 = la.norm(v1)
    n2 = la.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    else:
        return np.dot(v1/n1,v2/n2)

ords = dict([ (k,np.zeros(2048)) for k in targs ])

tups = [(2,100),(3,200),(4,400),(5,800)]
all_files = [ 'order_vectors_'+str(ng)+'_'+str(num)+'.pkl' for (ng,no) in tups for num in range(no) ]

def wait_and_save(ngram,csv_no,ords):
    outFile = 'order_vectors_'+str(ngram)+'_'+str(csv_no)+'.pkl'
    if all_files.index(outFile) > 0:
        waiting = True
        previous = all_files[all_files.index(outFile)-1]
    else:
        waiting = False
        previous = ''
    while waiting:
        sleep(30)
        if previous in os.listdir(home):
            waiting = False
    sleep(180)
    if previous:
        fh = open(home+previous,'r')
        prev = pickle.load(fh)
        for key in prev:
            ords[key] += prev[key]
        fh.close()
        com = 'rm ' + home + previous
        print "Removing previous order vectors"
        os.system(com)
    print "Saving outfile:", outFile
    fh = open(home+outFile,'w')
    pickle.dump(ords,fh)
    fh.close()

print "Beginning search through files for occurrences of target words..."
# http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-fiction-all-2gram-20090715-99.csv.zip
begin_time = time()
file = 'googlebooks-eng-fiction-all-'+str(ngram)+'gram-20090715-'+str(csv_no)+'.csv'
if file in os.listdir(tmp):
    print file
    for line in open(tmp+file,'r'):
        items = line.split()
        if len(items) < ngram+2:
            continue
        check = [word in targs for word in items[:ngram]]
        for i in range(len(check)):
            if check[i]:
                if all([ (word in env.keys() and not disallowed(word)) for word in items[0:i]+items[(i+1):ngram]]):
                    count = int(items[ngram+1])
                    ords[items[i]] += convolve_sequence(items[:ngram],i,count)
    print "Time this iteration:", str(time() - begin_time), "s"
    com = 'rm ' + tmp + file
    os.system(com)
    com = com + '.zip'
    os.system(com)
    wait_and_save(ngram,csv_no,ords)
else:
    print "File not found:", file


