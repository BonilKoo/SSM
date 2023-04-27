import time, os, itertools, math
from pysmiles import read_smiles
from copy import deepcopy
from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
# from scipy.special import softmax

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdm
from rdkit.Chem import AllChem, Draw, PandasTools, rdDepictor
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults
remover = SaltRemover()
rdDepictor.SetPreferCoordGen(True)

# avoid print warnings in 'pysmiles'
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

# read class
os.chdir('./srw_class')
#from set_device_seed import *   # seeding
from utils import *
from mychem import *            # RW functions
from mydata import *            # data

def _cal_prop_product(_n, _p):
    return np.array(_n * _p).astype(int)

# update (2022-04-05)
class DILInew():
    def __init__(self, chemistry = 'graph', n_rw=10, n_alpha = 0.5, iteration = 10, pruning=False, n_walker=100, rw_mode = "argmax", update_method = 1, ml_mode = 'train'):
        self.mode = ml_mode
        self.chemistry, self.n_rw, self.n_alpha = chemistry, n_rw, n_alpha
        self.n_iteration, self.pruning, self.n_walkers  = iteration, pruning, n_walker
        self.n_train, self.n_valid = 0, 0
        self.rw_mode = rw_mode
        #
        self.molinfo_df = pd.DataFrame()
        self.train_molinfo_df = pd.DataFrame()
        self.dTotalBondCount = dict()
        #
        self.lfraglist, self.ledgelist = [], []
        self.dEdgeUsedCount  = defaultdict(dict) # number of visited edges by RW for each graph; key: ltkbid, value: {edge_1:n_1, edge_2:n_2, ...}
        self.dEdgelistUsage  = defaultdict(dict)
        self.dNodeFragCount  = defaultdict(dict) # fragment dictionary for each molecule; key: ltkbid, value: {frag_1:n_1, frag_2:n_2, ...}
        self.dNodeFragSmiles = defaultdict(dict)
        #
        self.dEdgeClassDict = {} # summary for class
        #
        self.dNodeSubgraphCount = defaultdict(dict) # number of subgraphs to generate for each node in a graph; key: iteration {key: ltkbid, value: {node_1: n, node_2: n, ...}}
        self.dMolTransDict  = defaultdict(dict) # key: iteration, value: {ltkbid_0:T_0, ltkbid_1:T_1, ..., ltkbid_N:T_N}
        self.dMolPreferDict = defaultdict(dict) # key: iteration, value: {ltkbid_0:F_0, ltkbid_1:F_1, ..., ltkbid_N:F_N}
        self.dPreferDict = {} # key: iteration, value: Preference_iter
        self.dTrainLikelihood = {} # key: edge, 2nd key: iteration, value: ratio (tox/nontox)
        #
        self.lexclusivefrags = defaultdict(list)
        self.lunionfrags     = defaultdict(list)
        self.dFragSearch     = {}
        self.prunehistory    = defaultdict(dict)
        self.update_method   = update_method
        #
        # self.AtomAttentionDict = {} # key1: atom, key2: SMILES including 1st neighbor, key3: nIter, value: prob

    def getParams(self, get=False):
        print(f"RW: {self.n_rw}, Alpha: {self.n_alpha}, Initial T defined: {self.chemistry}")
        print(f"Transition mode: {self.rw_mode}, Pruning: {self.pruning}, Augmentation: {self.n_walkers}.")
        if get:
            return self.n_rw, self.n_alpha, self.chemistry, self.rw_mode, self.pruning, self.n_walkers

    def cal_total_bond_count(self):
        # chop-off all the graphs into single bonds and count them in terms of DILI labels. (1:19684, 0:15233 for dilist:train)
        self.dTotalBondCount = {i:float(sum([mol.GetNumBonds() for mol in self.molinfo_df.molobj[self.molinfo_df['class']==i]])) for i in self.molinfo_df['class'].unique()}

    def normalize_preference(self, _nIteration):
        _newfreq = pd.DataFrame(self.dEdgeClassDict[_nIteration]).transpose().div(self.dTotalBondCount, axis=1)
        self.dEdgeClassDict[_nIteration] = _newfreq.div(_newfreq.sum(axis=1), axis=0).fillna(0)

    def cal_nodeAttention(self, _nIter):
        for _molID in self.molinfo_df.index:
            nodelist = np.array(range(self.molinfo_df['molobj'][_molID].GetNumAtoms()))
            if _nIter == 0:
                self.dNodeSubgraphCount[_nIter][_molID] = dict(pd.Series([x for x in range(len(nodelist)) for y in range(self.n_walkers)]).value_counts())
                lMolAtomProbs = { i : 1 / self.molinfo_df["molobj"][_molID].GetNumAtoms() for i in range(self.molinfo_df["molobj"][_molID].GetNumAtoms()) }
            else:
                # bonds = [[x.GetBeginAtomIdx(), x.GetEndAtomIdx()] for x in self.molinfo_df['molobj'][_molID].GetBonds()]
                # for idx, bond in enumerate(bonds):
                for bond in self.molinfo_df['molobj'][_molID].GetBonds():
                    n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    bond_ind = bond.GetIdx()
                    # edge_smiles = Chem.MolFragmentToSmiles(self.molinfo_df['molobj'][_molID], atomsToUse=bond) # Jun14th
                    edge_smarts = Chem.MolFragmentToSmarts(self.molinfo_df['molobj'][_molID], atomsToUse=[n1, n2], bondsToUse = [bond_ind], isomericSmarts = True)
                    # print(_molID, edge_smiles, pd.Series(self.dEdgeClassDict[_nIter-1].loc[edge_smiles])) # done well
                    # Get Tox/NonTox Frequency values
                    try: vals = pd.Series(self.dEdgeClassDict[_nIter-1].loc[edge_smarts]) + 1e-10
                    except: vals = pd.Series([1,1])
                    # print(_molID, edge_smiles, vals) # done well
                    vals = vals.div(sum(vals))
                    entropy_score = 1 - sum(vals.apply(lambda x: -x*np.log2(x)))
                    # print(_molID, edge_smiles, entropy_score) # done well
                    self.molinfo_df['molgraph'][_molID].edges[n1,n2]['edge_smarts'] = edge_smarts
                    try: self.molinfo_df['molgraph'][_molID].edges[n1,n2]['entropy_score'][_nIter] = entropy_score
                    except:
                        self.molinfo_df['molgraph'][_molID].edges[n1,n2]['entropy_score'] = {}
                        self.molinfo_df['molgraph'][_molID].edges[n1,n2]['entropy_score'][_nIter] = entropy_score
                lMolAtomProbs = np.zeros(shape=self.molinfo_df["molobj"][_molID].GetNumAtoms())
                for node in self.molinfo_df["molgraph"][_molID].nodes():
                    nSum = 0
                    nDegree = self.molinfo_df["molgraph"][_molID].degree[node]
                    for x in self.molinfo_df["molgraph"][_molID].edges(node):
                        n1, n2 = x
                        nSum += self.molinfo_df["molgraph"][_molID].edges[n1,n2]['entropy_score'][_nIter]
                    nSum_deg = (nSum / np.sqrt(nDegree)) + 1e-5
                    lMolAtomProbs[node] = nSum_deg
                probs = lMolAtomProbs / np.array(lMolAtomProbs).sum()
#                 for idx, node in enumerate(nodelist):
#                     try: self.molinfo_df['molgraph'][_molID].nodes[node]['prob'][_nIter] = probs[idx]
#                     except:
#                         self.molinfo_df['molgraph'][_molID].nodes[node]['prob'] = {}
#                         self.molinfo_df['molgraph'][_molID].nodes[node]['prob'][_nIter] = probs[idx]

#                 counts = np.random.choice(nodelist, size = len(nodelist) * self.n_walkers, replace=True, p = probs)
                #
                counts = _cal_prop_product(len(nodelist) * self.n_walkers, probs)
                #
                resdict = {x:0 for x in range(len(nodelist))}
                resdict.update(dict(pd.Series(counts).value_counts()))
                self.dNodeSubgraphCount[_nIter][_molID] = resdict
#             # Record to attention dictionary
#             for node in self.molinfo_df.molobj[_molID].GetAtoms():
#                 lNeighborAtomList = [node.GetIdx()] + [x.GetIdx() for x in node.GetNeighbors()]
#                 lNeighborBondList = [x.GetIdx() for x in node.GetBonds()]
#                 smarts = Chem.MolFragmentToSmarts(self.molinfo_df.molobj[_molID],
#                                                  atomsToUse = lNeighborAtomList, bondsToUse = lNeighborBondList)
#                 try:
#                     self.AtomAttentionDict[node.GetSymbol()][smarts][_nIter].append(lMolAtomProbs[node])
#                 except:
#                     try: self.AtomAttentionDict[node.GetSymbol()][smarts][_nIter] = [lMolAtomProbs[node]]
#                     except:
#                         try: self.AtomAttentionDict[node.GetSymbol()][smarts] = {_nIter : [lMolAtomProbs[node]]}
#                         except:
#                             self.AtomAttentionDict[node.GetSymbol()] = {smarts:{_nIter : [lMolAtomProbs[node]]}}
    # END of cal_nodeAttention

    def DoRandomWalk(self, n_iter, ltkbid, T):  # run_rw starts here
        dpaths = mychem.cal_path_df(self.molinfo_df["molgraph"][ltkbid], T, n_walker = self.dNodeSubgraphCount[n_iter][ltkbid], walkLength = self.n_rw, mode = self.rw_mode )
        # dpaths: walklist(= list of nodes that a walker has gone through)
        #         {node1: visited_nodes, node2: visited_nodes}
        self.dEdgeUsedCount[n_iter][ltkbid], self.dEdgelistUsage[n_iter][ltkbid]    = mychem.rwr_summary(self.molinfo_df["molgraph"][ltkbid], dpaths, n_walker = self.dNodeSubgraphCount[n_iter][ltkbid])
        # dEdgeUsedCount: # of times edges used: {edge_1: 3, edge_2: 2, edge_3: 1, ...}
        # dEdgelistUsage: node: {node_list: [node_id, ..], edge_list: [edge_id, ..] }
        #                {0: {'node_list':[[0,1,2,,..], ..., []], 'edge_list': [['0_1', ...], ..., []]}
        self.dNodeFragCount[n_iter][ltkbid], self.dNodeFragSmiles[n_iter][ltkbid]   = mychem.rw_getSmilesPathDict(mychem, self.molinfo_df["molobj"][ltkbid], self.dEdgelistUsage[n_iter][ltkbid])
        # dNodeFragCount: {frag: 3, frag:1, ...}, dNodeFragSmiles = {node:frag, ... }
    # END of DoRandomWalk

    def cal_preference(self, n_it):
        for ltkbid in self.dEdgeUsedCount[n_it]:
            nClass = self.molinfo_df["class"][ltkbid]
            for edge in self.dEdgeUsedCount[n_it][ltkbid]:
                a,b = list(map(int, edge.split('_') ))
                bond_id = self.molinfo_df['molobj'][ltkbid].GetBondBetweenAtoms(a,b).GetIdx()
                frag = Chem.MolFragmentToSmarts(self.molinfo_df["molobj"][ltkbid], atomsToUse = [a, b], bondsToUse = [bond_id], isomericSmarts=True)
                try:
                    self.dEdgeClassDict[n_it][frag][nClass] += self.dEdgeUsedCount[n_it][ltkbid][edge]
                except:
                    try:
                        self.dEdgeClassDict[n_it][frag][nClass] = self.dEdgeUsedCount[n_it][ltkbid][edge]
                    except:
                        self.dEdgeClassDict[n_it][frag] = {}
                        self.dEdgeClassDict[n_it][frag][nClass] = self.dEdgeUsedCount[n_it][ltkbid][edge]
    # END of cal_preference

    def get_individual_F(self, n_iter, n_iter_pref_df, ltkbid, method = 1, mode=False):
        # get_individual_F(nI, pd_pref, ltkbid)
        # BEGIN of internal functions
        # Method 1.
        def get_likelihood_ratio(edge, mySeries, mode): # tox/nontox
            # mySeries from training data!
            mySeries = pd.Series(mySeries)
            probs = mySeries / mySeries.sum() + 1e-10
            if mode == 'test':
                try: val = self.dTrainLikelihood[edge][n_iter] # Use training data
                except: val = 1
            else: # train
                try:
                    val = self.dTrainLikelihood[edge][n_iter]
                except:
                    val = min( probs[1] / probs[0], 10)
                    try:    self.dTrainLikelihood[edge][n_iter] = val
                    except: self.dTrainLikelihood[edge] = {n_iter : val}
            return val

        # Method 2.
        def _get_likelihood(mySeries, mode): # class specific activation
            mySeries = pd.Series(mySeries)
            probs = mySeries / mySeries.sum() + 1e-10
            if mode == 'test': # test
                val = (1 - probs.div(probs.sum()).apply(lambda x: -x*math.log(x,2)).sum())
            else:
                val = min( probs[1] / probs[0], 10)
#             else: # train
#                 ind = self.train_molinfo_df["class"][ltkbid]
#                 myval = mySeries[ind] # WARNING FOR INDEX HERE!! Class: 1,2,3 vs INDEX: 0,1,2
#                 j = [x for k, x in enumerate(mySeries) if k!=ind]
#                 val = np.power(myval, len(j)) / ( np.prod(j) + 1e-8 )
            return val

        # Method 3.
        def _get_entropy_transition(mySeries, mode): # class specific activation
            mySeries = pd.Series(mySeries)
            probs = mySeries / mySeries.sum() + 1e-10
            val = (1 - probs.div(probs.sum()).apply(lambda x: -x*math.log(x,2)).sum())
            return val
        # Method 4.
        def get_likelihood_4(mySeries, mode):
            mySeries = pd.Series(mySeries)
            probs = mySeries / mySeries.sum() + 1e-10
            val = (1 - probs.div(probs.sum()).apply(lambda x: -x*math.log(x,2)).sum())
            return val

        def get_likelihood(mySeries, mode): # train/test same: entropy
            mySeries = pd.Series(mySeries)
            myval = mySeries / mySeries.sum()
            probs = pd.Series(myval + 1e-10)
            val = (1 - probs.div(probs.sum()).apply(lambda x: -x*math.log(x,2)).sum())
            return val
        # END of internal functions
        F = np.zeros( self.dMolTransDict[n_iter-1][ltkbid].shape )
        for edge in self.molinfo_df["molgraph"][ltkbid].edges():
            n1, n2 = edge
            if self.molinfo_df["molgraph"][ltkbid][n1][n2]['order'] != 0:
                bond = self.molinfo_df["molobj"][ltkbid].GetBondBetweenAtoms(n1, n2).GetIdx()
                # frag_smi = Chem.MolFragmentToSmiles(self.molinfo_df["molobj"][ltkbid], atomsToUse = [n1, n2], bondsToUse = [bond])
                frag_smi = Chem.MolFragmentToSmarts(self.molinfo_df['molobj'][ltkbid], atomsToUse = [n1, n2], bondsToUse = [bond], isomericSmarts = True)
                try:
                    try: probSeries = n_iter_pref_df[frag_smi] / n_iter_pref_df.sum(axis=1)
                    except: probSeries = n_iter_pref_df.T[frag_smi] / n_iter_pref_df.sum(axis=1)
                    if method == 1:
                        F[n1, n2] = get_likelihood_ratio(frag_smi, probSeries, mode=mode) # Method 1
                    elif method == 2:
                        F[n1, n2] = _get_likelihood( probSeries, mode=mode ) # Method 2
                    elif method == 3:
                        F[n1, n2] = _get_entropy_transition( mySeries, mode ) # Method 3
                    F[n2, n1] = F[n1, n2]
                except: F[n1, n2] = 0
        # normalize F
        for idx, row in enumerate(F):
            if row.sum() > 0:
                F[idx] = np.array([x/row.sum() for x in row]) # proportion
            else: continue
        F = F.transpose()
        return F
    # END of get_individual_F
    def rw_update_transitions(self, _T, _F, update_alpha):
        _T = _T * (1-update_alpha) + update_alpha * _F
        for idx, row in enumerate(_T):
            if row.sum() > 0:
                _T[idx] = np.array([x/row.sum() for x in row]) # proportion
        _T = _T.transpose()
        return _T
    # END of rw_update_transitions
    # START of get_fraglist
    def get_fraglist(self, n_iter):  # Get list of exclusive fragments
        #self.train_act_frag_df = pd.DataFrame(self.dNodeFragCount[n_iter], columns= self.dNodeFragCount[n_iter].keys())
        tempdf = pd.DataFrame(self.dNodeFragCount[n_iter], columns= self.dNodeFragCount[n_iter].keys()).fillna(0).T
        dfTempCnt = tempdf.merge(self.molinfo_df['class'], how='outer', left_index=True, right_on = self.molinfo_df.index)
        dfTempCnt.set_index(keys='key_0', inplace=True)
        nClassSpecificity  = dfTempCnt.groupby('class').any().astype(int).sum(axis=0)==1
        lExList = nClassSpecificity[nClassSpecificity].index.to_list()
        nClassSpecificity  = dfTempCnt.groupby('class').any().astype(int).sum(axis=0)>0
        lUnionList = nClassSpecificity[nClassSpecificity].index.to_list()
        return lExList, lUnionList
    # END of get_fraglist
    def search_fragments(self, sFrag):
        pdseries_search = pd.Series( 1e-10, index = self.train_molinfo_df['class'].unique(), name=sFrag)
        for ltkbid in self.train_molinfo_df["ID"]:
            if self.train_molinfo_df["molobj"][ltkbid].HasSubstructMatch(Chem.MolFromSmarts(sFrag), useChirality=True):
                pdseries_search[self.train_molinfo_df["class"][ltkbid]] += 1
        pdseries_search = pdseries_search.divide( self.train_molinfo_df.groupby("class")["ID"].count() + 1e-10 )
        return pdseries_search
    # END of search_fragments
    # START of DoPruning
    def DoPruning(self, n_iter, delta=0.05, pruning="pure", learnmode=False):
        # START of get_periedges
        def get_periedges(nxgraph): # get peripheral edges of the input subgraph
            res = []
            for sEdge in nxgraph.edges():
                n1, n2 = sEdge
                tempgraph = nxgraph.copy()
                tempgraph.remove_edge(n1, n2)
                tempobj = [x for x in nx.connected_components(tempgraph)]
                min_node_cnt = min([len(x) for x in tempobj])
                if (min_node_cnt == 1)  or (min_node_cnt == nxgraph.number_of_nodes()):
                    res.append(sEdge) # include only peripheral edges
            return res
        # END of get_periedges
        def calc_entropy(probs):
            probs = pd.Series(probs + 1e-10)
            return probs.div(probs.sum()).apply(lambda x: -x*math.log(x,2)).sum()
        def select_edge_to_remove(dscores): # when score = list of entropys
            frag = min(dscores.keys(), key=(lambda k: dscores[k]))
            minval = min(dscores[x] for x in dscores)
            return frag, minval
        def get_frag_subgraph(molgraph, paths):
            subgraph_edges = []
            for path in paths:
                a,b = list(map(int, path.split('_') ))
                subgraph_edges.append((a,b))
                subgraph_edges.append((b,a))
            nxSub = molgraph.edge_subgraph(subgraph_edges).copy()
            return nxSub
        def search_fragment_individual(frag_input):
            pdfragsearch = pd.Series(1e-10, index = self.train_molinfo_df['class'].unique(), name=frag_input)
            sSMARTS = Chem.MolFromSmarts(frag_input)
            for ltkbid in self.train_molinfo_df["ID"]:
                if self.train_molinfo_df["molobj"][ltkbid].HasSubstructMatch(sSMARTS, useChirality=True):
                    pdfragsearch[self.train_molinfo_df["class"][ltkbid]] += 1
            pdfragsearch = pdfragsearch.divide( self.train_molinfo_df.groupby("class")["ID"].count() + 1e-10, axis=0 )
            return pdfragsearch
        def while_condition(nEntNew, nEntOld, ltkbid, pdFragFreqClassNew, _nRatioOrg, method="pure", mode=learnmode):
            if mode==False: # train mode
                _nRatioOrg = self.train_molinfo_df['class'][ltkbid]
                ratio = (pdFragFreqClassNew[1] + 1e-10) / (pdFragFreqClassNew[0] + 1e-10) # class indication of train data
            else: # test mode
                try: ratio = (pdFragFreqClassNew[1]) / (pdFragFreqClassNew[0]) # class indication of train data
                except ZeroDivisionError: ratio = 100
            nOrgLR, nNewLR = np.log2(_nRatioOrg + 1e-10), np.log2(ratio + 1e-10) # Consistency test
            return (nEntNew < nEntOld) * (nOrgLR * nNewLR > 0)
        #
        # DoPruning Main starts here
        dHistory = defaultdict(dict)
        dFragHistory, dFragSearchCount, dEntropy = {}, {}, {}
        for ltkbid in self.molinfo_df["ID"]:
            nClass = self.molinfo_df["class"][ltkbid]
            dHistory[ltkbid] = {}  # key: ltkbid, subkey: nodeID
            nodes_to_pop = []
            for node in self.dNodeFragSmiles[n_iter][ltkbid]:
                dHistory[ltkbid][node] = {}
                #for i_walker in range(self.n_walkers):
                for i_walker in range(self.dNodeSubgraphCount[n_iter][ltkbid][node]):
                    # 1. Get Fragment SMILES
                    atomsOrg = self.dEdgelistUsage[n_iter][ltkbid][node]['node_list'][i_walker]
                    _, bondsOrg = mychem.rw_getatombondlist( self.molinfo_df["molobj"][ltkbid], self.dEdgelistUsage[n_iter][ltkbid][node]['edge_list'][i_walker] )
                    paths = self.dEdgelistUsage[n_iter][ltkbid][node]['edge_list'][i_walker]
                    nodeatom = self.molinfo_df["molobj"][ltkbid].GetAtomWithIdx(node).GetSymbol()
                    frag     = self.dNodeFragSmiles[n_iter][ltkbid][node][i_walker]
                    # 2. Count Fragment occurrence  # from self.dFragSearch[nI]
                    try: pdFragFreqClassOrg = self.dFragSearch[n_iter][frag]
                    except: pdFragFreqClassOrg = search_fragment_individual(frag)
                    try: nRatioOrg = ( pdFragFreqClassOrg[1]) / (pdFragFreqClassOrg[0]) # class indication of train data
                    except ZeroDivisionError: nRatioOrg = 100
                    # 3. initialize entrogpy
                    sFragOrg, nEntropyOrg, lPathListOrg = frag, calc_entropy( pdFragFreqClassOrg ), paths
                    dHistory[ltkbid][node][i_walker] = defaultdict(list)
                    atomsOld, bondsOld = deepcopy(atomsOrg), deepcopy(bondsOrg)
                    atomsNew, bondsNew = deepcopy(atomsOrg), deepcopy(bondsOrg)
                    sFragOld, nEntropyOld, lPathListOld, pdFragFreqClassOld, nAtomsOld = deepcopy(sFragOrg), deepcopy(nEntropyOrg)+1, deepcopy(lPathListOrg), deepcopy(pdFragFreqClassOrg), deepcopy(len(atomsOrg))
                    sFragNew, nEntropyNew, lPathListNew, pdFragFreqClassNew, nAtomsNew = deepcopy(sFragOrg), deepcopy(nEntropyOrg), deepcopy(lPathListOrg), deepcopy(pdFragFreqClassOrg), deepcopy(len(atomsOrg))
                    #
                    i_mod = 1
                    while while_condition(nEntropyNew, nEntropyOld, ltkbid, pdFragFreqClassNew, nRatioOrg, method = pruning, mode=learnmode): # Designed to follow the consistent decrease in entropy
                        # 0. initialize / update
                        atomsOld, bondsOld = deepcopy(atomsNew), deepcopy(bondsNew)
                        sFragOld, nEntropyOld, lPathListOld, pdFragFreqClassOld, nAtomsOld = deepcopy(sFragNew), deepcopy(nEntropyNew), deepcopy(lPathListNew), deepcopy(pdFragFreqClassNew), deepcopy(nAtomsNew)
                        dHistory[ltkbid][node][i_walker][i_mod] = [sFragOld, nEntropyOld, pdFragFreqClassOld]
                        # 1. get subgraph information:
                        nxSubgraph = get_frag_subgraph(self.molinfo_df["molgraph"][ltkbid], lPathListOld)
                        if (len(nxSubgraph.edges()) == 1):
                            if (i_mod == 1):
                                dHistory[ltkbid][node][i_walker][i_mod].append('Input_singlet')
                                dFragHistory[sFragOld] = dHistory[ltkbid][node][i_walker]
                                break
                            else:
                                dHistory[ltkbid][node][i_walker][i_mod].append('Final_singlet')
                                dFragHistory[sFragOld] = dHistory[ltkbid][node][i_walker]
                                break
                        else: pass
                        # 2. get peripheral edges / non-breakable edges
                        lPeriEdges = get_periedges(nxSubgraph)
                        # 3. get argmin edges
                        dPeriScore, dPeriClassDict, dPeriPath, dAtoms, dBonds = {}, {}, {}, {}, {}
                        for sPeriEdge in lPeriEdges:
                            a,b = sPeriEdge
                            topop = [str(a)+"_"+str(b), str(b)+"_"+str(a)]
                            lPathList_i = [x for x in lPathListOld if not x in topop]
                            atoms, bonds = mychem.rw_getatombondlist( self.molinfo_df["molobj"][ltkbid], lPathList_i )
                            # try: sFrag_i  = Chem.MolFragmentToSmiles( self.molinfo_df["molobj"][ltkbid], atomsToUse = atoms, bondsToUse = bonds)
                            try: sFrag_i = Chem.MolFragmentToSmarts(self.molinfo_df['molobj'][ltkbid], atomsToUse = atoms, bondsToUse = bonds, isomericSmarts=True)
                            except:
                                # sFrag_i = Chem.MolFragmentToSmiles(self.molinfo_df["molobj"][ltkbid], atomsToUse = atoms)
                                sFrag_i = Chem.MolFragmentToSmarts(self.molinfo_df['molobj'][ltkbid], atomsToUse = atoms, isomericSmarts=True)
                                print(f"Warning: (In pruning), there are missing edges in {self.molinfo_df['smiles'][ltkbid]} between atoms: {atoms}")
                            try: dTempFreq = dFragSearchCount[sFrag_i]
                            except:
                                dTempFreq = search_fragment_individual(sFrag_i)
                                dFragSearchCount[sFrag_i] = dTempFreq
                            try: nEntropyTemp = dEntropy[sFrag_i]
                            except:
                                nEntropyTemp = calc_entropy( dTempFreq )
                                dEntropy[sFrag_i] = nEntropyTemp
                            dPeriScore[sFrag_i] = nEntropyTemp
                            dPeriClassDict[sFrag_i] = dTempFreq
                            dPeriPath[sFrag_i] = lPathList_i
                            dAtoms[sFrag_i] = atoms
                            dBonds[sFrag_i] = bonds
                        sFragTemp, nEntropyTemp = select_edge_to_remove(dPeriScore)
                        pdFragFreqClassTemp     = dPeriClassDict[sFragTemp]
                        lPathListTemp           = dPeriPath[sFragTemp]
                        nAtomsTemp              = len(dAtoms[sFragTemp])
                        atomsTemp, bondsTemp    = dAtoms[sFragTemp], dBonds[sFragTemp]
                        atomsNew, bondsNew = atomsTemp, bondsTemp
                        sFragNew, nEntropyNew, lPathListNew, pdFragFreqClassNew, nAtomsNew = sFragTemp, nEntropyTemp, lPathListTemp, pdFragFreqClassTemp, nAtomsTemp
                        dHistory[ltkbid][node][i_walker][i_mod].append('continued')
                        #
                        i_mod += 1
                    else: # while loop
                        if i_mod > 1:
                            i_mod -= 1 # while else
                            dHistory[ltkbid][node][i_walker][i_mod].append('No_improvement_in_entropy')
                        else:
                            dHistory[ltkbid][node][i_walker][i_mod] = [sFragOld, nEntropyOld, pdFragFreqClassOld, 'Entropy not satisfied at first stage']
                        dFragHistory[sFragOld] = dHistory[ltkbid][node][i_walker]
                    # END of while loop
                    # print(dHistory[ltkbid][node][i_walker])   ### check validity
                    if sFragOrg != sFragOld:
                        #     self.dSRW[nI]  =  [ dNodeFragSmiles, dNodeFragCount, dEdgeUsedCount, dEdgelistUsage ]
                        #     dNodeFragSmiles = {node:frag, ... }
                        #     dNodeFragCount: {frag: 3, frag:1, ...},
                        #     dEdgeUsedCount: edge_id : used_count  ==  {0: 3, 1: 2, 2: 1, 3: 1, 4: 3, 5: 2, 7: 2, 8: 1}
                        #     dEdgelistUsage: node: {node_list: [node_id, ..], edge_list: [edge_id, ..], usage: n_used/n_edges, nxpath_list=['1_2', '2_3', '3_4',...]}
                        # 4. remove old (exclusive) fragment "Old tag"
                        self.dNodeFragSmiles[n_iter][ltkbid][node][i_walker] = sFragOld
                        self.dNodeFragCount[n_iter][ltkbid][sFragOrg] -= 1
                        try:    self.dNodeFragCount[n_iter][ltkbid][sFragOld] += 1
                        except: self.dNodeFragCount[n_iter][ltkbid][sFragOld]  = 1
                        bondpoplist = list( set(lPathListOrg) - set(lPathListOld)  )
                        for bondpop in bondpoplist:
                            self.dEdgeUsedCount[n_iter][ltkbid][bondpop] -= 1
                        self.dEdgelistUsage[n_iter][ltkbid][node]['node_list'][i_walker] = atomsOld
                        self.dEdgelistUsage[n_iter][ltkbid][node]['edge_list'][i_walker] = lPathListOld
            # END: FOR node in self.dNodeFragSmiles[n_iter][ltkbid]:
            for node in nodes_to_pop:
                self.dNodeFragSmiles[n_iter][ltkbid].pop(node)
                self.dEdgelistUsage[n_iter][ltkbid].pop(node)
        # END of for: ltkbid
        pdHistory = pd.DataFrame(dHistory, columns = dHistory.keys())
        return pdHistory
    #END of DoPruning
    # BEGIN of subgraph_split()
    # def subgraph_split(self, method=False):

    # MAIN HERE - argument: train_duata
    def train(self, train_data):
        self.molinfo_df, self.train_molinfo_df = train_data, train_data
        self.n_train = train_data.shape[0]
        self.cal_total_bond_count()
        print(f'The Number of allowed walks: {self.n_rw}')
        for nI in range(self.n_iteration):  # iterate random walk process
            start = time.time()
            print(nI, 'loop starts', end="")
            self.cal_nodeAttention(nI) # calculate the number of subgraphs for each graph given number_of_walkers
            if nI > 0:
                pd_pref =  pd.DataFrame( self.dEdgeClassDict[nI-1], columns = self.dEdgeClassDict[nI-1].keys() ).fillna(0).T
            for ltkbid in self.molinfo_df["ID"]: # iterate over molecules
                smiles = self.molinfo_df["smiles"][ltkbid]
                if nI == 0: # cal_T(molobj, molgraph, smiles, chemistry='graph')
                    T = mychem.cal_T(mychem, self.molinfo_df["molobj"][ltkbid], self.molinfo_df["molgraph"][ltkbid], smiles, chemistry = self.chemistry)
                else:
                    self.dMolPreferDict[nI-1][ltkbid] = self.get_individual_F(nI, pd_pref, ltkbid, method = self.update_method)
                    T = self.rw_update_transitions(self.dMolTransDict[nI-1][ltkbid], self.dMolPreferDict[nI-1][ltkbid], self.n_alpha) # T * (1-alpha) + F * alpha
                self.dMolTransDict[nI][ltkbid] = T
                self.DoRandomWalk(nI, ltkbid, T) # each molecule
            self.dEdgeClassDict[nI] = {}
            self.cal_preference(nI) # dEdgeClassDict: Save Preference each iteration
            self.lexclusivefrags[nI], self.lunionfrags[nI] = self.get_fraglist(nI)
            if self.pruning:
                self.dFragSearch[nI] = pd.Series(self.lunionfrags[nI]).apply(self.search_fragments)
                self.dFragSearch[nI].index = self.lunionfrags[nI]
                self.dFragSearch[nI] = self.dFragSearch[nI].T
                self.prunehistory[nI] = self.DoPruning(nI, delta=0.05, pruning = self.pruning )
                self.dEdgeClassDict[nI] = {}
                self.cal_preference(nI) # Save Preference each iteration
                self.lexclusivefrags[nI], self.lunionfrags[nI] = self.get_fraglist(nI)
            self.normalize_preference(nI) # normalize dEdgeClassDict
            fin = round( ( time.time() - start ) / 60 , 3 )
            print(f' for Random Walk {self.n_rw} completed in {fin} mins.')
        return self.dEdgeClassDict
    # END of train
    # MAIN HERE - argument: valid_data
    def valid(self, valid_df, train_df, train_obj):
        #### Read Train data
        self.molinfo_df = valid_df
        self.train_molinfo_df = train_df
        self.dEdgeClassDict = train_obj.dEdgeClassDict
        train_dFragSearch = train_obj.dFragSearch
        ####
        self.n_valid = self.molinfo_df.shape[0]
        print(f'The Number of allowed walks: {self.n_rw}')
        for nI in range(self.n_iteration):  # iterate random walk process
            start = time.time()
            print(nI, 'loop starts', end="")
            self.cal_nodeAttention(nI) # calculate the number of subgraphs for each graph given number_of_walkers
            if nI > 0:
                pd_pref =  pd.DataFrame( self.dEdgeClassDict[nI-1], columns = self.dEdgeClassDict[nI-1].keys() ).fillna(0).T # Load From train data
            for ltkbid in self.molinfo_df.index: # iterate over molecules
                smiles = self.molinfo_df["smiles"][ltkbid]
                if nI == 0:
                    # cal_T(molobj, molgraph, smiles, chemistry='graph')
                    T = mychem.cal_T(mychem, self.molinfo_df["molobj"][ltkbid], self.molinfo_df["molgraph"][ltkbid], smiles, chemistry = self.chemistry)
                else:
                    self.dMolPreferDict[nI-1][ltkbid] = self.get_individual_F(nI, pd_pref, ltkbid, method = self.update_method, mode='test')
                    T = self.rw_update_transitions(self.dMolTransDict[nI-1][ltkbid], self.dMolPreferDict[nI-1][ltkbid], self.n_alpha) # T * (1-alpha) + F * alpha
                self.dMolTransDict[nI][ltkbid] = T
                self.DoRandomWalk(nI, ltkbid, T) # each molecule
            if self.pruning:
                self.dFragSearch[nI] = train_dFragSearch[nI].T
                self.prunehistory[nI] = self.DoPruning(nI, delta=0.05, pruning = self.pruning, learnmode='test' )
            fin = round( ( time.time() - start ) / 60 , 3 )
            print(f' for Random Walk {self.n_rw} completed in {fin} mins.')


class analyze_individual():
    def __init__(self):
        self.frag_df = defaultdict(pd.DataFrame)
        self.lfraglist = defaultdict(list)
        self.ledgelist = defaultdict(list)
    def get_frag_df(self, srw, iteration=0):
        self.frag_df[iteration] = pd.DataFrame(srw.dNodeFragCount[iteration], columns= srw.dNodeFragCount[iteration].keys())
        self.frag_df[iteration] = self.frag_df[iteration].fillna(0).T
        self.lfraglist[iteration] = list(set(self.frag_df[iteration].columns))
        print(f'The number of fragments for iteration {iteration}: {len(self.lfraglist[iteration])}')
    def get_edge_df(self, iteration=0):
        self.ledgelist = list(set(self.dEdgeClassDict[iteration].keys()))
        print(f'The number of edges for iteration {iteration}: {len(self.ledgelist[iteration])}')

def prepare_classification(df, molinfo, binarize=False):
    if binarize:
        df_binary = df > 0
    else:
        df_binary = df
    df_binary = df_binary.astype(int)
    df_new = df.merge(molinfo['class'], how='outer', left_index=True, right_on=molinfo["ID"])
    df_new = df_new.set_index("key_0", drop=True)
    df_new.index.name = None
    df_new = df_new.fillna(0)
    X = df_new.drop('class',axis=1)
    y = df_new['class']
    return X, y
