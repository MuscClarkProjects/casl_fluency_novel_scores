'''
Create an object that reads in a pPMI file and allows computation of
cosine similarities, generation of adjacency matrices on the basis
of cosim, etc.
'''
import json
import pickle
import math
import numpy as np
from numpy.linalg import norm

class Cosimilator():
    def __init__(self, pmi_f):
        with open(pmi_f,'r') as f:
            if pmi_f.endswith("json"):
                pmi = json.load(f)
            else:
                pmi = pickle.load(f)

        self.targets = pmi.keys()
        self.contexts = []
        for t in pmi.keys():
            self.contexts.extend(pmi[t].keys())
        self.contexts = list(set(self.contexts))
        self.counts = pmi
        self.norms = dict([(t,self.norm(self.counts[t])) for t in self.targets])

    def norm(self,v):
        return math.sqrt(sum([ val**2 for val in v.values()]))

    def dotprod(self,v1,v2):
        eN = set(self.counts[v1].keys()).intersection(self.counts[v2].keys())
        return sum([ self.counts[v1][k] * self.counts[v2][k] for k in eN ])

    def cosine_similarity(self,v1,v2):
        try:
            denom = self.norms[v1] * self.norms[v2]
            if denom != 0:
                return self.dotprod(v1,v2)/denom
            else: return 0.0
        except KeyError:
            print "Missing one of these:", v1, v2
            return 0.0

    def threshold(self,n,theta):
        return 1.0 if n >= theta else 0.0

    def threshold_distance(self,n,theta):
        return 1.0 if n <= theta else 0.0

    def create_adjacency_matrix(self,wordlol=None,funk=None,z_thresh=1.0,abs_dist_thresh=None):
        ''' Accepts a list of lists of words, a comparison function, and Z-score
            threshold for considering words to be "linked". Returns a hash table with
            word pairs as keys and values of either 1 (linked) or 0.
        '''
        if not funk:
            funk = self.cosine_similarity
        if not wordlol:
            wordlol = [self.targets]
        cartesian = []
        for lol in wordlol:
            cartesian.extend(set([ (w1,w2) for w1 in lol for w2 in lol if w1 < w2 ]))
        self.all_cosims = dict([ ((w1,w2),funk(w1,w2)) for (w1,w2) in cartesian ])
        if not abs_dist_thresh:
            cvs = [ cv for cv in self.all_cosims.values() if not np.isnan(cv) ]
            # print cvs
            mu = np.mean(cvs)
            print mu
            sig = np.std(cvs)
            print sig
            self.all_zs = [ (words,(self.all_cosims[words]-mu)/sig) for words in self.all_cosims ]
            self.adj = dict([ (words,self.threshold(zed,z_thresh)) for (words,zed) in self.all_zs ])
        else:
            self.adj = dict([ (words,self.threshold_distance(dist,abs_dist_thresh)) for (words,dist) in self.all_cosims ])
        percent_above_thresh = sum(self.adj.values())/len(self.adj)
        print "Percentage selected: ", percent_above_thresh
        return self.adj

    def transition_matrix(self,el):
        ''' Construct transition matrix = sum of outer products of context vectors for
            consecutive words in list el
        '''
        cons = []
        for t in el:
            cons.extend(self.counts[t].keys())
        cons = sorted(list(set(cons)))
        bigs = self.bigrams(el)
        tmat = np.zeros((len(cons),len(cons)))
        for (pre,post) in bigs:
            prevec = np.zeros((1,len(cons)))
            postvec = np.zeros((1,len(cons)))
            for (i,con) in enumerate(cons):
                if self.counts[pre].has_key(con):
                    prevec[0,i] += self.counts[pre][con]
                if self.counts[post].has_key(con):
                    postvec[0,i] += self.counts[post][con]
            tmat += np.outer(prevec,postvec)
        return tmat

    def transition_vector(self,el):
        cons = []
        for t in el:
            cons.extend(self.counts[t].keys())
        cons = sorted(list(set(cons)))
        bigs = self.bigrams(el)
        tvec = np.zeros((1,len(cons)))
        for (pre,post) in bigs:
            prevec = np.zeros((1,len(cons)))
            postvec = np.zeros((1,len(cons)))
            for (i,con) in enumerate(cons):
                if self.counts[pre].has_key(con):
                    prevec[0,i] += self.counts[pre][con]
                if self.counts[post].has_key(con):
                    postvec[0,i] += self.counts[post][con]
            tvec += np.multiply(prevec,postvec)
        return tvec

    def bigrams(self,el):
        return [ (el[i],el[i+1]) for i in range(len(el)-1) ]

    def transition_norm(self,el,transition=None,order=2):
        if transition == None:
            transition = self.transition_matrix(el)
        return norm(transition,ord=order)
