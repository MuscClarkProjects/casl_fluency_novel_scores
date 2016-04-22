#----------------------------------#
#    Metrics for Verbal Fluency    #
#----------------------------------#

import re
from nltk import *
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
import numpy as np
from numpy.linalg import svd, eig, eigh
# import sgt
from itertools import ifilterfalse
from random import shuffle, random
import networkx as nx
import cmath

# Raw score
# Take clean list and count lines that do not begin with
# a special character (i.e., #, %, !, -, etc.) Return count
def raw_score(lst):
    lst = [ 1 for el in lst if el[0] not in [ '#', '!', '-', '?', '@' ] ]
    return sum(lst)

# Repetitions
# Take a raw list and count lines that begin with #
def repetitions(lst,label='#'):
    lst = [ 1 for el in lst if el[0] == label ]
    return sum(lst)

def exponential_proximity(d,k = -0.34657):
    return np.exp(k * d)

def fractional_proximity(d):
    return 1.0/(1+d)

# Relative efficiency
# Function takes (1) a list of items and (2) a function for computing efficiency 
# on the metric (semantic, phonological, or orthographic similarity)
# Must measure efficiency of original list and on up to 999 permutations of the
# list. Returns a number between 0 and 1 (ranking of original order over total
# number of permutations evaluated).
# This flatten is FAST, but flattens even tuples!
def flatten(lol):
    return [item for sublist in lol for item in sublist]

def flatten_lists(lol):
    # Leave tuples intact
    flat = []
    for el in lol:
        for item in el:
            flat.append(item)
    return flat

def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

def fact(n):
    if n == 1: return 1
    else:
        return n * fact(n-1)

def heterogeneous_permutations(lst,max_perm=10000):
    cache = set([])
    cache.add(tuple(lst))
    if len(lst) < 8:
        max_perm = fact(len(lst))
    while len(cache) < max_perm:
        while tuple(lst) in cache:
            shuffle(lst)
        cache.add(tuple(lst))
        yield lst
    else:
        raise StopIteration

def memoize(function):
    '''A nice memoize function, taken from 
    the internet'''
    cache = {}
    def decorated_function(*args):
        if args in cache:
            return cache[args]
        else:
            val = function(*args)
            cache[args] = val
            return val
    return decorated_function

@memoize
def semantic_similarity(tup1,tup2):
    syn1 = wn.synsets(tup1[0])[tup1[1]]
    syn2 = wn.synsets(tup2[0])[tup2[1]]
    return wn.path_similarity(syn1,syn2)

def special_maximum(lst):
    if lst:
       return max(lst)
    else: return 0

def allcorrespondences(s1,s2):
    if re.match('^\s*$',s1) or not s2:
        return 0
    try:
        d = s1.index(s2[0])
        return (1.0/(d+1)) + allcorrespondences(s1[:d]+s1[(d+1):],s2[1:])
    except ValueError:
        return allcorrespondences(s1[1:],s2[1:])

def remove_characters(orig,unwanted):
    bild = []
    for c in orig:
        if c not in unwanted:
            bild.append(c)
    return ''.join(bild)

@memoize
def string_overlap(s1,s2):
    s1 = remove_characters(s1,"-_'")
    s2 = remove_characters(s2,"-_'")
    if len(s2) < len(s1):
        temp = s1
        s1 = s2
        s2 = temp
    if len(s1)==0 and len(s2)==0: return 1.0
    lendif = len(s2) - len(s1)
    denom = len(s2)
    mx = 0
    for step in range(1+lendif):
        nu = (' ' * step) + s1 + (' ' * (lendif-step))
        numer = allcorrespondences(nu,s2)
        over = numer/denom
        if over > mx:
            mx = over
    return mx

@memoize
def edit_distance(s1,s2,remove_punk=False):
    if remove_punk:
        s1 = remove_characters(s1,"-_'")
        s2 = remove_characters(s2,"-_'")
    if s1 == '': return len(s2)
    elif s2 == '': return len(s1)
    else:
        ins_w1 = 1 + edit_distance(s1[:-1],s2)
        ins_w2 = 1 + edit_distance(s1,s2[:-1])
        diag = edit_distance(s1[:-1],s2[:-1])
        if s1[-1] != s2[-1]: diag += 2
    ed = min([ins_w1,ins_w2,diag])
    return ed

def edit_proximity(s1,s2):
    s1 = remove_characters(s1,"-_'")
    s2 = remove_characters(s2,"-_'")
    ed = edit_distance(s1,s2)
    totalen = len(s1) + len(s2)
    if totalen == 0: return 1.0
    else:
        return 1.0 - (float(ed)/(totalen))
    
# For each repetition, award a fraction of a point depending on
# how close it is to the first time the subject said the word
# This should actually use a synset... don't want goose and geese 
# not to be considered a repetition.
def repetition_distances(lst):
    # use tuples with position, item
    reps = [ (ix[0],ix[1][1:].strip()) for ix in enumerate(lst) if ix[1][0] == '#' ]
    total = 0
    for rep in reps:
        overlaps = [ string_overlap(el[1],rep[1]) for el in enumerate(lst) if el[0] < rep[0] ]
        orig = overlaps.index(max(overlaps))
        total += (1.0/(rep[0]-orig))
    return total

entries = cmudict.entries()
cmu = dict(entries)


def norm_array(ar):
    nrm = np.sqrt(np.dot(ar,ar))
    return ar/nrm


def remove_punk(lst):
    res = []
    for el in lst:
        res.extend([ m for m in re.findall('[A-Za-z_\-]*',el) if m != '' ])
    return res


# @memoize
def phonemic_edit_proximity(s1,s2):
    """ Takes two lists of phonemes and calculates overlap
    between them using the edit distance algorithm.
    """
    # Remove accent information
    remove_accent = lambda phon: ''.join([ c for c in phon if ord(c) in range(65,91) ])
    el1 = map(remove_accent, s1)
    el2 = map(remove_accent, s2)
    # List all phonemes
    all_phonemes = list(set(el1 + el2))
    # map each phoneme to an arbitrary character
    arb = dict(zip(all_phonemes,[ chr(65+x) for x in range(len(all_phonemes)) ]))
    w1 = ''.join([ arb[ph] for ph in el1 ])
    w2 = ''.join([ arb[ph] for ph in el2 ])
    return edit_proximity(w1,w2)

def list_bigrams(lst,dist=1,bookends=True):
    # print lst
    if bookends: lst = ['<s>'] + lst + ['</s>']
    all_bigrams = [ tuple([el0,el1,abs(i0-i1)]) for (i0,el0) in enumerate(lst) for (i1,el1) in enumerate(lst) if el0<el1 ]
    return list(set([ (t[0],t[1]) for t in all_bigrams if t[2] <= dist ]))

def internal(lst):
    if len(lst) < 3:
        return []
    else: return lst[1:-1]

def purge_lists(lop,lol):
    '''if anything from the list of purgers (lop) is in a list in lol, remove the list from lol'''
    return [ el for el in lol if not any([ item in el for item in lop ]) ]

def the_other_one(item,tup):
    if tup[0] == item: return tup[1]
    else: return tup[0]

def deprecated_optimize_list(lst,comparison=edit_proximity):
    # Make a list of all possible bigrams, but use indices within list instead of actual items
    if not lst:
        return []
    bigrams = list_bigrams(range(len(lst)),dist=len(lst),bookends=False)
    interm = sorted([ [b[0],b[1],comparison(lst[b[0]],lst[b[1]])] for b in bigrams ],key=lambda v: v[2])
    master = interm.pop()[0:2]
    while len(master) < len(lst):
        remnant = [ i for i in range(len(lst)) if i not in master ]
        edges = [ i for i in interm if (master[0] in i[0:2] or master[-1] in i[0:2]) and any([ r in i[0:2] for r in remnant ]) ]
        best = edges[-1]
        if master[0] in best[0:2]:
            master.insert(0,the_other_one(master[0],best))
        else:
            master.append(the_other_one(master[-1],best))
    if sorted(master) != range(len(lst)):
        print "Error in optimize_list"
        try:
            print "Optimized:", [ lst[i] for i in master ]
        except IndexError:
            print master
        print "Original:", lst
        print [ (word,len([ w for w in lst if w == word ])) for word in lst ]
        exit()
    # convert master back to items from lst
    opt = [ lst[i] for i in master ]
    return opt

def optimize_list(lst,comparison=edit_proximity):
    if len(lst) <= 2: return lst
    edges = sorted([ (comparison(w1,w2),w1,w2) for w1 in lst for w2 in lst if w1 < w2 ],reverse=True)
    bildin = [ edges[0][1], edges[0][2] ]
    for i in range(len(lst)):
        earlies = sorted([ (c,w2) for (c,w1,w2) in edges if w1 == bildin[0] and w2 not in bildin] \
                       + [ (c,w1) for (c,w1,w2) in edges if w2 == bildin[0] and w1 not in bildin],reverse=True)
        laters = sorted([ (c,w2) for (c,w1,w2) in edges if w1 == bildin[-1] and w2 not in bildin] \
                       + [ (c,w1) for (c,w1,w2) in edges if w2 == bildin[-1] and w1 not in bildin],reverse=True)
        if not earlies and not laters:
            return bildin
        if not laters or earlies[0] > laters[0]:
            bildin = [earlies[0][1]] + bildin
        elif not earlies or laters[0] >= earlies[0]:
            bildin.append(laters[0][1])
    return bildin

def make_graph_for_pbil(lst,comparison=edit_proximity):
    """
    Create a graph that has three weights per edge.
    a(dj) = 0 : is the edge included in a Hamiltonian path?
    p(rob) = 2/(n*(n-1)), where n is the length of the list. Prob of choosing edge for current path candidates
    s(im) = comparison function on vertices incident to edges
    """
    g = nx.Graph()
    n = len(lst)
    pinit = 2.0/(n * (n-1))
    edgy = [ (w1,w2,{'a':0,'p':pinit,'s':comparison(w1,w2)}) for w1 in lst for w2 in lst if w1 < w2 ]
    g.add_weighted_edges_from(edgy)
    return g

def make_cdf_for_pbil(g):
    e = [ (w1,w2,d['weight']['p']) for (w1,w2,d) in g.edges(data=True) ]
    cdf = [ (e[i][0],e[i][1],sum([ e[j][2] for j in range(i) ])) for i in range(len(e)) ]
    return cdf

def select_edge_from_cdf(cdf):
    r = random()
    threshed = [ c for c in cdf if c[2] < r ]
    return threshed[-1]

def degree_tuples(g,key='a'):
    return [ (node,sum([ e[2]['weight'][key] for e in g.edges(data=True)])) for node in g.nodes() if e[0] == node or e[1] == node ]

def renormalize_graph(g):
    degrees = dict(degree_tuples(g))
    print degrees
    nunodes = [ n for n in g.nodes() if degrees[n] < 2 ]
    if len(nunodes) > 2:
        nug = Graph()
        edgy = [ (w1,w2,d) for (w1,w2,d) in g.edges() if w1 in nunodes or w2 in nunodes ]
        denom = sum([ e[2]['weight']['p'] for e in edgy ])
        nug.add_weighted_edges_from([ (w1,w2,{'a':d['weight']['a'],'p':d['weight']['p']/denom,'s':d['s']}) for (w1,w2,d) in edgy ])
        return nug
    return None
        
def derive_new_path(g):
    for i in range(len(g.nodes())-1):
        cdf = make_cdf_for_pbil(g)
        (w1,w2,d) = select_edge_from_cdf(cdf)
        g[w1][w2]['weight']['a'] = 1.0
        g = renormalize_graph(g)
    deg = sorted(degree_tuples(g),key=lambda (x,y),(p,q): y < q)
    edgy = g.edges(data=True)
    path = [e for e in edgy if e[0] == deg[0][0]]
    sims = []
    while edgy:
        for e in edgy:
            if e[0] in path[-1]:
                next_edge = e
                next_node = e[1]
                break
            elif e[1] in path[-1]:
                next_edge = e
                next_node = e[0]
                break
        path.extend(next_edge)
        edgy.remove(next_edge)
        sims.append(next_edge[2]['weight']['s'])
    return (sum(sims),path)              
        

def coherence(lst,comparison=edit_proximity):
    if len(lst) < 2:
        return 0.0
    elif len(lst) == 2: return 1.0
    bigrams = list_bigrams(lst,bookends=False)
    similarity = lambda bigram: comparison(bigram[0],bigram[1])
    sim = sum(map(similarity,bigrams))
    opt = optimize_list(lst,comparison=comparison)
    bigrams = list_bigrams(opt,bookends=False)
    return sim/sum(map(similarity,bigrams))

def deprecated_coherence(lst,comparison=string_overlap):
    if len(lst) < 7: permutor = permutations
    else: permutor = heterogeneous_permutations
    bigrams = list_bigrams(lst,bookends=False)
    similarity = lambda bigram: comparison(bigram[0],bigram[1])
    sim = sum(map(similarity,bigrams))
    sims = []
    for perm in permutor(lst):
        bigrams = list_bigrams(perm,bookends=False)
        sims.append(sum(map(similarity,bigrams)))
    sims.append(sim)
    sims = sorted(sims)
    return sims.index(sim)/float(len(sims))

# Semantic, orthographic, or phonemic range
# Find the semantic similarity of all pairs of items in the list
# Subtract the lowest value from the highest value
def metric_range(lst,comparison=edit_proximity):
    bigrams = list_bigrams(lst,dist=len(lst),bookends=False)
    similarity = lambda bigram: comparison(bigram[0],bigram[1])
    sims = map(similarity,bigrams)
    if not sims:
        return 0.0
    mx = max(sims)
    mn = min(sims)
    return mx-mn

# Persistence
# Take a raw list and look for the '--30' mark
# If not there return None
# If there, split the list at that point and return (# of valid items after mark)/(total # of valid items)
def persistence(raw):
    try:
        indy = raw.index('--30')
    except ValueError:
        return None
    pre = [ d for d in raw[:indy] if d[0] not in [ '!', '#', '@', '-', '?', '(' ] ]
    post = [ d for d in raw[indy:] if d[0] not in [ '!', '#', '@', '-', '?', '(' ] ]
    return float(len(post))/(len(pre) + len(post))

# Markov model probability
# First construct a Markov model: requires hash of clean lists
#     The Markov model consists of (1) a hash of list elements -> probabilities (or log probs) 
# Second, compute probability of list: requires Markov model from step 1.
# Third, estimate all unigram probabilities
# Fourth, compute probability of list of bigrams based on the products of unigram freqs.
# Fifth, return log((Prod of bigram probabilities)/(Prod of bigram probabilities estimated in step 4))
def make_markov(clean,left_out={}):
    # count all occurring words (unigrams)
    uni_C = {}
    uni_T = 0
    all_words = flatten_lists([clean[key] for key in clean]) + (len(clean.keys()) * ['<s>','</s>'])
    left_out_words = flatten_lists([left_out[key] for key in left_out if left_out[key] not in all_words])
    # Compute unigram frequencies using all words, including those from "left_out"
    for word in all_words+left_out_words:
        uni_T += 1
        try:
            uni_C[word] += 1.0
        except KeyError:
            uni_C[word] = 1.0
    for key in uni_C.keys():
        uni_C[key] = uni_C[key] / uni_T
    # count occurrences of bigrams that actually occur
    bi_C = {}
    bigrams = flatten_lists([list_bigrams(clean[key]) for key in clean]) # These bigrams occur in training set only
    for bigram in bigrams:
        try:
            bi_C[bigram] += 1.0
        except KeyError:
            bi_C[bigram] = 1
    # Get smoothed probabilities of bigrams that occurred in "clean"
    if left_out:
        (markov,discount) = sgt.simpleGoodTuringProbs(bi_C)
    else:
        markov = {}
        total_C = sum(bi_C.values())
        for bi in bi_C:
            markov[bi] = bi_C[bi]/total_C
        discount = 0.0
    # compute joint probability of all possible bigrams
    all_bigrams = [ (u1,u2) for u1 in uni_C.keys() for u2 in uni_C.keys() if u1 < u2 ]
    joint_probs = {}
    sum_prob = 0
    for bigram in all_bigrams:
        joint_probs[bigram] = uni_C[bigram[0]] * uni_C[bigram[1]]
        if not markov.has_key(bigram):
            sum_prob += joint_probs[bigram]
    # Give each unseen bigram a portion of the discounted probability, in proportion
    # to its joint probability (computed as the product of unigram frequencies)
    unseen = flatten([list_bigrams(left_out[key]) for key in left_out]) # bigrams for "test" set only
    for key in unseen:
        if not markov.has_key(key):
            markov[key] = (joint_probs[bigram]/sum_prob) * discount
    return (markov, joint_probs)

def markov_score(lst,markov,joint):
    bigrams = list_bigrams(lst)
    numer = sum([log(markov[key]) for key in bigrams ])
    denom = sum([log(joint[key]) for key in bigrams ])
    return (numer - denom)


def create_giant_matrix(master,mat_map):
    m = len(master) ** 2
    n = len(mat_map.keys())
    giant = zeros((m,n))
    for (co,dude) in enumerate(mat_map):
        giant[:,co] = reshape(mat_map[dude],m,order='F')
    return giant

# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# An algorithm to compute PCA. Not as fast as the NumPy implementation
from pylab import *

def pca(data,nRedDim=0,normalise=1):
    
    # Centre data
    m = mean(data,axis=0)
    data -= m

    # Covariance matrix
    C = cov(transpose(data))

    # Compute eigenvalues and sort into descending order
    evals,evecs = linalg.eig(C) 
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]

    if nRedDim>0:
        evecs = evecs[:,:nRedDim]
    
    if normalise:
        for i in range(shape(evecs)[1]):
            evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])

    # Produce the new data matrix
    x = dot(transpose(evecs),transpose(data))
    # Compute the original data again
    y=transpose(dot(evecs,x))+m
    return x,y,evals,evecs

def compute_pca_scores(giant,mat_map,how_many=None):
    (x,y,evals,evecs) = pca(giant)
    # Reconstruct data with each evec,
    # then compute dot product of patient
    # matrix with reconstructed matrix to
    # get a score.
    scores = {}
    if not how_many:
        how_many = len(evecs)
    for evecol in range(how_many):
        evec = evecs[:,evecol] # Should be size (46,1)
        recon = dot(transpose(evec),transpose(giant)) # The dot prod is (1,~30000)
        escore = dot(recon,giant) # Should be 46 scores
        for (num,dude) in enumerate(mat_map):
            try:
                scores[dude].append(escore[num])
            except KeyError:
                scores[dude] = [escore[num]]
    return scores
            
def compute_ica_scores(giant,ics,mat_map,how_many=None):
    # Reconstruct data with each IC
    # 
    scores = {}
    if not how_many:
        how_many = ics.shape[1]
    for ic_no in range(how_many):
        ic = ics[:,ic_no]
        escore = dot(ic.T,giant) # Should be 67 scores
        for (num,dude) in enumerate(mat_map): # What is the guarantee that these dudes come out in the same order as in the escore vector?
            try:
                scores[dude].append(escore[num])
            except KeyError:
                scores[dude] = [escore[num]]
    return scores

def average_paths(g):
    if nx.is_connected(g):
        aspl = nx.average_shortest_path_length(g,weighted=True)
    return aspl

# This is meant to work on connected graphs. If the
# graph is not connected, it will return None.
def algebraic_connectivity_of_graph(graph):
    if not graph:
        return 0.0
    ls = sorted(nx.laplacian_spectrum(graph))
    # Is the graph connected?
    if (ls[1] - ls[0]) > 1e-10:
        return ls[1]
    else: return 0.0

# This version acts on the (potentially weighted) adjacency matrix
def algebraic_connectivity(adj):
    d = adj.sum(axis=0)
    D = diag(d)
    L = D - adj
    (ls,evex) = eig(L)
    return ls[1]

# Here is algebraic connectivity using a complex matrix
def complex_algebraic_connectivity(cadj):
    if cadj is not None:
        (m,n) = cadj.shape
        if m > 1:
            (ls,evex) = eigh(cadj)
            reals = [ el.real for el in ls ]
            reals.sort()
            if len(reals) > 1:
                return reals[1]
    return 0

# Minmax normalization for interval from -1 to 1 for complex algebraic connectivity
def minmax(el,flur=-1):
    if len(el) == 0:
        return el
    mn = min(el)
    mx = max(el)
    denom = mx-mn
    if denom != 0:
        mm = [ (2.0*abs(flur) * (x - mn))/denom + flur for x in el ]
    else:
        mm = el
    return mm

# Add magnitudes of polar complex numbers
def summag(el):
    return sum([ cmath.polar(c)[0] for c in el ])

# This takes a graph with weights that are distances
def graph_radius(graph):
    sp = nx.shortest_path_length(graph,weight='weight')
    ecc = nx.eccentricity(graph,sp=sp)
    if ecc:
        rad = nx.radius(graph,e=ecc)
    else:
        rad = 0
    return rad

# This takes a graph with weights that are distances
def graph_diameter(graph):
    sp = nx.shortest_path_length(graph,weight='weight')
    ecc = nx.eccentricity(graph,sp=sp)
    if ecc:
        dia = nx.diameter(graph,e=ecc)
    else:
        dia = 0
    return dia
