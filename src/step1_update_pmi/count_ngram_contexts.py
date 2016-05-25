import os
from re import match
from string import lower
import pickle
from time import time, sleep
from sys import argv
from random import random

tmp = '/ifs/tmp/'
home = '/ifshome/dclark/ngrams/'
# home = '/Users/dgclark/Documents/dcfiles/manuscripts/new_easton/ngrams/'
present = os.listdir(tmp)

ngrams = argv[1]
lower = argv[2]
upper = argv[3]

# Load stop words
with open('../../data/stopwords/stopWords.txt', 'r') as fh:
    stops = fh.read().strip().split(',')


def disallowed(w):
    return any([letter in w for letter in '!@#$%^&*(){}1234567890=+[]'])


def updateHash(items, n, hash):
    ''' Take a line from an n-grams file,
        the number of n-grams per line, and a dictionary.
        Return an updated version of the dictionary with
        the count of the context word(s) added.
    '''
    if len(items) < n or items[0] in stops or items[n-1] in stops:
        return hash

    segments = ([ (0,i) for i in range(1,n) ], [ (i,n) for i in range(1,n) ])
    for boundary_type in (0, 1):
        context_index = n - 1 if boundary_types == 0 else 0

        if disallowed(items[context_index]):
            return hash

        for (start, end) in segments[boundary_type]:
            test = '_'.join(items[start:end])
            if hash.has_key(test):
                try:
                    hash[test][boundary_type][items[context_index]] += int(items[n+1])
                except KeyError:
                    hash[test][boundary_type][items[context_index]] = int(items[n+1])
    return hash


# Load target words
# Format of target hash:
#    targets[<target word>][0 for context word to left of target, 1 for context word to R][<context word>] = <count of context word occurrences>
with open(home+'targetWords.pkl','r') as fh:
    targs = pickle.load(fh)

targs.extend(['semiconscious','semi_-_conscious','semiannual','semi_-_annual','mahimahi','mahi_-_mahi','blackeyed_peas',\
              'black_-_eyed_peas','anticommunist','anti_-_communist','threetoed_sloth','three_-_toed_sloth',\
              'twotoed_sloth','two_-_toed_sloth','black_-_eyed_pea','blackeyed_pea','twotoed_sloths','two_-_toed_sloths',\
              'threetoed_sloths','three_-_toed_sloths','anticommunists','anti_-_communists','mahi_-_mahis','mahimahis'])
targets = dict([ (t.lower(),({},{})) for t in targs ])


toget = [ str(i) for i in range(int(lower),int(upper)) ]
for fileno in toget:
    print fileno
    start = time()
    present = os.listdir(tmp)
    filename = 'googlebooks-eng-all-'+ngrams+'gram-20090715-' + fileno + '.csv.zip'
    success = False
    while not success:
        if filename[:-4] not in present:
            sleep(1)
	    # com = 'unzip ' + tmp + filename + ' -d ' + tmp
	    # os.system(com)
    	try:
            fh = open(tmp+filename[:-4],'r')
            success = True
            fh.close()
    	except IOError:
            sleep(90)
    for line in open(tmp+filename[:-4],'r'):
        try:
            items = [ thing.lower() for thing in line.split() ]
    	except IndexError:
            continue
        targets = updateHash(items, int(ngrams), targets)
    com = 'rm ' + tmp + filename
    print com
    os.system(com)
    com = 'rm ' + tmp + filename[:-4]
    os.system(com)
    p = open(home+'collocations_'+ngrams+'_'+lower+'_'+upper+'.pkl','w')
    pickle.dump(targets,p)
    p.close()
    print "Elapsed time this iteration = ", time() - start
