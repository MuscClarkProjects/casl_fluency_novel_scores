"""
Generate cluster and switch scores for letter fluency, simplified program for new analysis
"""

import json
import networkx as nx
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import isnan, array, corrcoef
import numpy.linalg as la
from os import listdir, system
import pickle
import re

from cosimilator import Cosimilator
from metrix import *
from vf_list import VF_list


def data_directory(sub_dir):
    return os.path.join('../../data/', sub_dir)


def step1_directory(sub_dir):
    return data_directory(os.path.join('step1', sub_dir))


def step2_directory(sub_dir):
    return data_directory(os.path.join('step2', sub_dir))


list_directory = step1_directory('lists_lowercase/')


subcategories_directory = step1_directory('subcategories/')


def pronunciations_f(f):
    return os.path.join(step1_directory('pronunciations/'), f)


def subcategories_f(f):
    return os.path.join(step1_directory('subcategories/'), f)


def order(tup):
    return tuple(sorted(tup))


def same_initial_letters(w1,w2):
    if w1[0:2] == w2[0:2]:
        return True
    else: return False


def homonyms(w1,w2,pronunciations):
    return pronunciations[w1] == pronunciations[w2]


def rhyme(w1,w2,pronunciations):
    """Two words rhyme if they end with identical pronunciations and the
    identical parts contain an accented vowel"""
    p1 = pronunciations[w1]
    p2 = pronunciations[w2]
    shorter = lambda x,y: x if len(x) < len(y) else y
    rime = []
    # Is there any part of the endings that is identical
    for loc in range(1,len(shorter(p1,p2))):
        if p1[-loc] != p2[-loc]:
            break
        else:
            rime.append(p1[-loc])
    if not rime: return False

    def is_vowel(w):
        return w[0] in 'AEIOU' and (w[-1].isdigit() and int(w[-1]) > 0)

    vowels = [ el for el in rime if is_vowel(el) ]
    return len(vowels) > 0


def one_vowel_difference(w1,w2,pronunciations):
    p1 = pronunciations[w1]
    p2 = pronunciations[w2]
    diffs = [ sound for sound in set(p1 + p2) if (sound in p1 and not sound in p2) or (sound in p2 and not sound in p1) ]
    if len(diffs) == 2 and diffs[0][0] in 'AEIOU' and diffs[1][0] in 'AEIOU':
        return True
    else: return False

wnl = WordNetLemmatizer()


def lemmatize(tup):
    """This uses the tuple from the spelling to synset map"""
    try:
        s = wn.synsets(tup[0])[tup[1]]
        n = s.name.split('.')[0]
    except IndexError:
        n = tup[0]
    return wnl.lemmatize(n)


def semantic_similarity(tup1,tup2):
    synset1 = wn.synsets(tup1[0])[tup1[1]]
    synset2 = wn.synsets(tup2[0])[tup2[1]]
    return synset1.path_similarity(synset2)


def remove_punk_and_inflections(lst,s2m):
    punk = [ '#', '!', '(' ]
    words = []
    for word in lst:
        if word[0] in punk:
            words.append(word[1:].strip())
        elif word[0] != '-':
            words.append(word.strip())
    try:
        clean = [ lemmatize(s2m[word]) for word in words ]
        orig = [ wnl.lemmatize(word) for word in words ]
    except KeyError:
        print lst, words
        exit()
    return (clean,orig)


def cosine_similarity(v1,v2):
    """ Returns cosine similarity of two row vectors from PMI matrix """
    n1 = v1/np.sqrt(np.dot(v1,v1.T))
    n2 = v2/np.sqrt(np.dot(v2,v2.T))
    return np.dot(n1,n2.T)


def predict(values,coefs):
    """Takes n values and n+1 coefs (including intercept) and returns
    a probability from a logistic regression model"""
    values = [1] + values
    total = -1 * sum([ v*c for (v,c) in zip(values,coefs) ])
    logistic = 1.0/(1.0+np.exp(total))
    return logistic


def identify1cluster(item, lst, adj):
    """item: a word in the list, lst: a list of words, adj: a hash table,
    s.t., you put in an ordered pair of words and get back a 1 if they are
    linked, else 0. Returns a list of contiguous words linked to item"""
    no = lst.index(item)
    preds = [lst[i] for i in range(no-1, -1, -1)]
    succs = [lst[i] for i in range(no+1, len(lst))]
    clust = [item]
    for pred in preds:
        ordered = order([pred, item])
        if not adj.has_key(ordered):
            adj[ordered] = 0
        if adj[ordered]:
            clust.append(pred)
        else:
            break
    for succ in succs:
        ordered_succ = order([succ, item])

        if not adj.has_key(ordered_succ):
            adj[ordered_succ] = 0
        if adj[ordered_succ]:
            clust.append(succ)
        else:
            break
    return clust


def filter_out_cluster_subsets(lol):
    """lol is a list of lists, with each list being a cluster. This function
    compares each cluster to all the others and removes it from the lol
    if it is a subset of any other cluster"""
    clusters = []
    los = [ set(el) for el in lol ]
    for current_set in los:
        # Is lst already in clusters?
        cluster_sets = [ c for c in clusters ]
        supersets = [ c for c in cluster_sets if c.issuperset(current_set) ]
        if not supersets: # i.e., if the current set is not contained in any cluster already added
            subsets = [ c for c in cluster_sets if c.issubset(current_set) ]
            for subset in subsets:
                clusters.remove(subset)
            clusters.append(current_set)
    return [ list(c) for c in clusters ]


def load_subcategories(f):
    with open(f,'r') as fh:
        data = fh.readlines()

    adj = {}
    comp = []
    for item in data:
        (sub,vals) = item.strip().split(' : ')
        comp.extend(vals)
        keys = set([ (v1,v2) for v1 in vals.split(',') for v2 in vals.split(',') if v1 < v2 ])
        for (v1,v2) in keys:
            adj[(v1,v2)] = 1
    non_linked = set([ (c1,c2) for c1 in comp for c2 in comp if c1 < c2 ])
    for nl in non_linked:
        if not adj.has_key(nl):
            adj[nl] = 0
    return adj


def read_pronunciations():
    with open(pronunciations_f('master_pronunciations.csv'), 'r') as fh:
        data = fh.readlines()

    data = [ d.split(',') for d in data ]
    return {w.strip(): pronun.strip().split('.') for (w, pronun) in data}


def update_pronunciations(lop,pronuns):
    for p in lop:
        nup = raw_input('Enter pronunciation for ' + p + ': ')
        pronuns[p] = nup.split('.')
    fh = open(pronunciations_f(task+'_pronunciations.txt'),'w')
    for w in pronuns:
        fh.write(w + ' : ')
        fh.write('.'.join(pronuns[w]))
        fh.write('\n')
    fh.close()
    return pronuns


class Letter_adjacency():
    def __init__(self,s2p):
        self.adj = {}
        self.pronuns = s2p
        for (w1,w2) in [ (w1,w2) for w1 in self.pronuns for w2 in self.pronuns if w1 < w2 ]:
            if same_initial_letters(w1,w2) or one_vowel_difference(w1,w2,self.pronuns) or homonyms(w1,w2,self.pronuns) or rhyme(w1,w2,self.pronuns):
                self.adj[(w1,w2)] = 1.0
            else:
                self.adj[(w1,w2)] = 0.0

    def __getitem__(self,(w1,w2)):
        if w1 > w2:
            (w2,w1) = (w1,w2)
        if self.adj.has_key((w1,w2)):
            return self.adj[(w1,w2)]
        else:
            return 0.0

    def has_key(self,key):
        return self.adj.has_key(key)

    def __setitem__(self,key,val):
        self.adj[key] = val

    def __repr__(self):
        rep = "{"
        for k in self.adj.keys():
            rep += str(k) + " : " + str(self.adj[k]) + ", "
        return rep + "}"


def identify_clusters(lst, adj):
    """lst: a list of words, adj: a hash table (as above). Returns
    a list of lists, each of which is a cluster"""
    clusters = []
    for item in lst:
        clusters.append(identify1cluster(item, lst, adj))
    return filter_out_cluster_subsets(clusters)


def cluster_and_switch_scores(clusters):
    switches = len(clusters)-1.0
    if len(clusters) == 0:
        cluster_size = 0
        switches = 0
    else:
        switches = len(clusters) - 1.0
        cluster_size = sum([ len(clu)-1.0 for clu in clusters ])/len(clusters)
    return (cluster_size,switches)


def protected_logarithm(n):
    if n <=0: return 0.0
    else: return np.log(n)


with open(step1_directory('correctedLogUnigrams.pkl'),'r') as fh:
    freqs = pickle.load(fh)


def get_word_locations():
    with open(step2_directory('word_locations.json'), 'r') as wl:
        word_locations = json.load(wl)
    return word_locations


def raw_list_to_zscores(f_name, task, word_locations=get_word_locations()):
    words = word_locations[task]
    score = None
    for (word, file_locations) in words.iteritems():
        f_indexes = [f['index'] for f in file_locations if f['file'] == f_name]
        if len(f_indexes) == 0:
            continue

        other_indexes = [f['index'] for f in file_locations
            if f['file'] != f_name]
        if len(other_indexes) < 2:
            continue

        other_mean = np.mean(other_indexes)
        other_std = np.std(other_indexes)

        curr_score = sum([(i - other_mean)/other_std for i in f_indexes])
        score = curr_score + score if score is not None else curr_score

    return score


# Maybe put all the header items into a hash table as keys. Each value can then
# be a function that takes the fluency file name as an argument and returns
# a string for printing into the file. The string can be unprocessed data (age, sex),
# or the result of some measurement on the fluency list.

class DataTable():
    def __init__(self, task, master_cosimilator):
        # Get the file names
        self.files = [ f for f in listdir(list_directory)
            if re.search('CASL(\d+)_y\d_' + task + '\.txt$', f)]
        self.ids = [ re.findall('CASL(\d+)_', f)[0] for f in self.files ]
        self.visits = [ re.findall('CASL\d+_(y\d)_', f)[0] for f in self.files]
        self.indices = dict([(f, num) for (num, f) in enumerate(self.files)])
        self.pronuns = read_pronunciations()
        print("loaded pronuns")
        self.pfmemo = {}
        # Perform traditional cluster/switch measurements
        if task in ['animals','fruits_and_veg']:
            task_f = subcategories_directory + '/' + task+'_subcategories.txt'
            self.adj = load_subcategories(task_f)
        else:
            self.adj = Letter_adjacency(self.pronuns)
        print("loaded subcategories")
        self.vfs = [ VF_list(list_directory+f) for f in self.files ]
        self.valids = [ vf.valid() for vf in self.vfs ]
        print("loaded valids")
        self.cands = self._cluster_and_switch()
        # Perform semantic cluster and switch a la Pakhomov
        self.cosim = master_cosimilator
        cos_adj = self.cosim.create_adjacency_matrix(wordlol=self.valids)
        print("loaded adj matrix")
        self.pakhomov = self._cluster_and_switch(cos_adj)
        print("calculated pakhomov")
        # Perform phonological and orthographic cluster and switch a la Pakhomov
        # self.phon_funk = lambda w1,w2: dpo(w1,w2,pronunciations = self.pronuns)
        phon_adj = self.cosim.create_adjacency_matrix(wordlol=self.valids, funk=self._phon_funk)
        self.phono_cs = self._cluster_and_switch(phon_adj)
        print("calculated phono_cs")
        ortho_adj = self.cosim.create_adjacency_matrix(wordlol=self.valids,funk=edit_proximity)
        self.ortho_cs = self._cluster_and_switch(ortho_adj)
        self.semantic_graphs = [ self._make_graph(vf,cos_adj) for vf in self.valids ]
        self.phono_graphs = [ self._make_graph(vf,phon_adj) for vf in self.valids ]
        self.ortho_graphs = [ self._make_graph(vf,ortho_adj) for vf in self.valids ]
        print("calculated semantic, phono, and ortho graphs")
        self.transitions = {}
        self._task = task
        self._word_locations = get_word_locations()

    def _phon_funk(self,w1,w2):
        p1 = self.pronuns[w1]
        p2 = self.pronuns[w2]
        if w1 < w2:
            kee = (w1,w2)
        else:
            kee = (w2,w1)
        try:
            return self.pfmemo[kee]
        except KeyError:
            self.pfmemo[kee] = phonemic_edit_proximity(p1,p2)
        except AttributeError:
            for ix in [ x for x in dir(self) ]:
                print ix
                exit()
        return self.pfmemo[kee]

    def _cluster_and_switch(self,adj=None):
        if not adj:
            adj = self.adj
        c_and_s = [ cluster_and_switch_scores(identify_clusters(v, adj)) for v in self.valids ]
        return c_and_s

    def _bigrams(self,wordlist):
        return [ (w1,w2) for w1 in wordlist for w2 in wordlist if w1 < w2 ]

    def _make_weighted_graph(self,f,funk,distance = False):
        if distance:
            func = lambda w1,w2: 1.0 - funk(w1,w2)
        else: func = funk
        vals = self.valids[self.indices[f]]
        biggy = self._bigrams(vals)
        edges = [ (w1,w2,func(w1,w2)) for (w1,w2) in biggy ]
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        return g

    def _make_graph(self,vf,adj):
        g = nx.Graph()
        g.add_edges_from([ (w1,w2) for w1 in vf for w2 in vf if w1 < w2 and adj[(w1,w2)] == 1 ])
        return g

    def _list_method_strings(self):
        unordered = [m for m in dir(self) if m[0] != '_' and callable(getattr(self,m))]
        return ['id'] + [m for m in unordered if m != 'id']

    def _list_score_methods(self):
        return [ getattr(self,m) for m in self._list_method_strings()]

    def _format_result(self,r):
        return r if isinstance(r,str) else str(r)

    def _run_all_methods(self,f):
        ''' Run each score method and return a string with the scores,
            delimited by tabs.
        '''
        metherds = self._list_score_methods()
        return [ self._format_result(m(f)) for m in metherds ]

    def _mean_values(self,d):
        v = d.values()
        return mean(v) if v else 0.0

    def _max(self,d):
        v = d.values()
        return max(v) if v else 0.0

    def id(self,f):
        return self.ids[self.indices[f]]

    def visit(self, f):
        return self.visits[self.indices[f]]

    def task(self,f):
        return self._task

    def raw(self,f):
        return self.vfs[self.indices[f]].raw()

    def repetitions(self,f):
        return self.vfs[self.indices[f]].reps()

    def intrusions(self,f):
        return self.vfs[self.indices[f]].intrusions()

    def sum_frequency(self,f):
        val = self.vfs[self.indices[f]].valid()
        return sum([ freqs[v] for v in val ])

    def sum_recip_freq(self,f):
        val = self.vfs[self.indices[f]].valid()
        return sum([ 1.0/freqs[v] for v in val if freqs[v] > 0 ])

    def mean_frequency(self,f):
        val = self.vfs[self.indices[f]].valid()
        return mean([ freqs[v] for v in val ]) if len(val) > 0 else 0.0

    def sum_syllables(self,f):
        val = self.vfs[self.indices[f]].valid()
        missing = [w for w in val if not self.pronuns.has_key(w)]
        if missing:
            self.pronuns = update_pronunciations(missing,self.pronuns)
        return sum([len([p for p in self.pronuns[w] if p[0] in 'AEIOU']) for w in val])

    def mean_syllables(self,f):
        val = self.vfs[self.indices[f]].valid()
        missing = [w for w in val if not self.pronuns.has_key(w)]
        if missing:
            self.pronuns = update_pronunciations(missing,self.pronuns)
        return mean([len([p for p in self.pronuns[w] if p[0] in 'AEIOU']) for w in val])

    def _transition_matrix(self,f):
        if self.transitions.has_key(f):
            return self.transitions[f]
        else:
            self.transitions[f] = self.cosim.transition_matrix(self.vfs[self.indices[f]].valid())
            return self.transitions[f]

    def cluster(self,f):
        return self.cands[self.indices[f]][0]

    def switch(self,f):
        return self.cands[self.indices[f]][1]

    def sem_cluster(self,f):
        return self.pakhomov[self.indices[f]][0]

    def sem_switch(self,f):
        return self.pakhomov[self.indices[f]][1]

    def phon_cluster(self,f):
        return self.phono_cs[self.indices[f]][0]

    def phon_switch(self,f):
        return self.phono_cs[self.indices[f]][1]

    def ortho_cluster(self,f):
        return self.ortho_cs[self.indices[f]][0]

    def ortho_switch(self,f):
        return self.ortho_cs[self.indices[f]][1]

    def sem_alg_con(self,f):
        return algebraic_connectivity_of_graph(self._make_weighted_graph(f,self.cosim.cosine_similarity))

    def phon_alg_con(self,f):
        return algebraic_connectivity_of_graph(self._make_weighted_graph(f,self._phon_funk))

    def ortho_alg_con(self,f):
        return algebraic_connectivity_of_graph(self._make_weighted_graph(f,edit_proximity))

    def sem_coherence(self,f,funk = None):
        if funk is None:
            funk = self.cosim.cosine_similarity
        return coherence(self.valids[self.indices[f]],comparison=funk)

    def phon_coherence(self,f):
        return self.sem_coherence(f,funk = self._phon_funk)

    def ortho_coherence(self,f):
        return self.sem_coherence(f,funk = edit_proximity)

    def mr_sem(self,f,funk = None):
        if funk is None:
            funk = self.cosim.cosine_similarity
        return metric_range(self.valids[self.indices[f]],comparison=funk)

    def mr_phon(self,f):
        return self.mr_sem(f,funk = self._phon_funk)

    def mr_ortho(self,f):
        return self.mr_sem(f,funk = edit_proximity)

    def mr_freq(self,f):
        return self.mr_sem(f,funk = lambda w1,w2: abs(freqs[w1]-freqs[w2]))

    def sem_radius(self,f,funk = None):
        if not funk:
            funk = self.cosim.cosine_similarity
        return graph_radius(self._make_weighted_graph(f,funk,distance=True))

    def phon_radius(self,f):
        return self.sem_radius(f,funk=self._phon_funk)

    def ortho_radius(self,f):
        return self.sem_radius(f,funk=edit_proximity)

    def sem_diameter(self,f,funk = None):
        if not funk:
            funk = self.cosim.cosine_similarity
        try:
            return graph_diameter(self._make_weighted_graph(f,funk,distance=True))
        except ValueError:
            print self.valids[self.indices[f]]
            return -1

    def phon_diameter(self,f):
        return self.sem_diameter(f,funk=self._phon_funk)

    def ortho_diameter(self,f):
        return self.sem_diameter(f,funk=edit_proximity)

    def sem_average_degree(self,f,graph=None):
        if not graph:
            graph = self.semantic_graphs[self.indices[f]]
        if len(graph.edges()) > 0:
            return self._mean_values(graph.degree())
        else: return 0.0

    def ortho_average_degree(self,f):
        return self.sem_average_degree(f,graph=self.ortho_graphs[self.indices[f]])

    def phono_average_degree(self,f):
        return self.sem_average_degree(f,graph=self.phono_graphs[self.indices[f]])

    def sem_average_cluster_coeff(self,f):
        return self._mean_values(nx.clustering(self.semantic_graphs[self.indices[f]]))

    def phono_average_cluster_coeff(self,f):
        return self._mean_values(nx.clustering(self.phono_graphs[self.indices[f]]))

    def ortho_average_cluster_coeff(self,f):
        return self._mean_values(nx.clustering(self.ortho_graphs[self.indices[f]]))

    def sem_transitivity(self,f):
        return nx.transitivity(self.semantic_graphs[self.indices[f]])

    def phono_transitivity(self,f):
        return nx.transitivity(self.phono_graphs[self.indices[f]])

    def ortho_transitivity(self,f):
        return nx.transitivity(self.ortho_graphs[self.indices[f]])

    def sem_max_betweenness(self,f):
        return self._max(nx.betweenness_centrality(self.semantic_graphs[self.indices[f]]))

    def ortho_max_betweenness(self,f):
        return self._max(nx.betweenness_centrality(self.ortho_graphs[self.indices[f]]))

    def phono_max_betweenness(self,f):
        return self._max(nx.betweenness_centrality(self.phono_graphs[self.indices[f]]))

    def index_z_score(self, f):
        return raw_list_to_zscores(f, self._task, self._word_locations)


#tasks=['a', 'animals', 'boats', 'f', 'fruits_and_veg', 's', 'tools',
# 'vehicles', 'verbs', 'water_creatures']
def write_tasks(pmi_f, tasks=['animals'], dest_dir='../data/step2/'):
    # Write all of the DataTable method calls to a file
    master_cosimilator = Cosimilator(pmi_f) if isinstance(pmi_f, str) else pmi_f
    print "Done assembling cosimilator."
    for task in tasks:
        print task
        dt = DataTable(task, master_cosimilator)

        methods = dt._list_score_methods()
        method_names = dt._list_method_strings()
        header = '\t'.join(method_names) # dt._list_score_methods())

        outfile_name = os.path.join('../../data/step2/', 'CASL_' + task + '.txt')
        with open(outfile_name, 'w') as outfile:
            outfile.write(header)
            for f in dt.files:
                print f
                outfile.write('\n')
                results = '\t'.join([dt._format_result(m(f)) for m in methods])
                outfile.write(results)
