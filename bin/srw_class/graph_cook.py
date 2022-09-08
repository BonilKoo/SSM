import os, sys, time, itertools, random

import rdkit
from rdkit import Chem

import pandas as pd
import numpy as np
import networkx as nx
from multiprocessing import Pool

from scipy.sparse.linalg.eigen.arpack import eigsh

#
def my_entropy(probs):
    probs = pd.Series(probs) + 1e-10
    return probs.div(probs.sum()).apply(lambda x: -x*np.log(x)).sum()
#
################################################################################
class mol_obj():
    def __init__(self, smi):
        import networkx as nx
        self.smi = smi # subgraph smiles
        self.mol = Chem.MolFromSmarts(smi)
        self.molentropy = 0.0
        self.nodelist = [x.GetIdx() for x in self.mol.GetAtoms()]
        self.candidates = []
        self.candidateSmiles = [] # subgraph smiles set
        self.dSubgraphSmiles = {} # key: nodeset, value: [smiles1, smiles2]
        self.dFreq = {} # key: smiles, value: [freq_tox, freq_nontox]
        self.dEntropy = {} 
        self.dIG = {}   # key: [s1, s2], value: IG
        self.datagraphs = '' # dataframe
    
    def division(self, selected=False):
        # enumerate node list
        for nodesize in range(1,len(self.nodelist)):
            SubgraphNodeList = [set(x) for x in itertools.combinations(self.nodelist, nodesize)]
            self.candidates += SubgraphNodeList
        for nodelist in self.candidates:
            subgraph = Chem.MolFragmentToSmiles(self.mol, atomsToUse = list(nodelist))
            if '.' not in subgraph:
                self.candidateSmiles.append(subgraph)
        self.candidateSmiles = list(set(self.candidateSmiles))
        # generate subgraphs with more than single atom sized
        for nodeset_1 in self.candidates:
            nodeset_1 = list(nodeset_1)
            nodeset_2 = list(set(self.nodelist) - set(nodeset_1))
            subgraph_1 = Chem.MolFragmentToSmiles( self.mol, atomsToUse = nodeset_1 )
            subgraph_2 = Chem.MolFragmentToSmiles( self.mol, atomsToUse = nodeset_2 )
            if ('.' not in subgraph_1) and ('.' not in subgraph_2):
                self.dSubgraphSmiles[tuple(nodeset_1)] = [subgraph_1, subgraph_2]

    def cal_entropy(self, subgraph, SRWobj):
        nNonTox, nTox = SRWobj.molinfo_df['class'].value_counts()[0], SRWobj.molinfo_df['class'].value_counts()[1]
        freq_Tox = sum([x.HasSubstructMatch(Chem.MolFromSmarts(subgraph)) for x in SRWobj.molinfo_df.molobj[SRWobj.molinfo_df['class']==1]]) / nTox
        freq_NonTox = sum([x.HasSubstructMatch(Chem.MolFromSmarts(subgraph)) for x in SRWobj.molinfo_df.molobj[SRWobj.molinfo_df['class']==0]]) / nNonTox
        self.dFreq[subgraph] = [freq_Tox, freq_NonTox]
        self.dEntropy[subgraph] = my_entropy(self.dFreq[subgraph])

    def GetMolEntropy(self, SRWobj):
        nTox, nNonTox = SRWobj.molinfo_df['class'].value_counts()
        freq_Tox = sum([x.HasSubstructMatch(Chem.MolFromSmarts(self.smi)) for x in SRWobj.molinfo_df.molobj[SRWobj.molinfo_df['class']==1]]) / nTox
        freq_NonTox = sum([x.HasSubstructMatch(Chem.MolFromSmarts(self.smi)) for x in SRWobj.molinfo_df.molobj[SRWobj.molinfo_df['class']==0]]) / nNonTox
        self.molentropy = my_entropy([freq_Tox, freq_NonTox])

    def cal_information_gain(self):
        mol_n = self.mol.GetNumAtoms()
        # IG = mol_entropy - { (sub1_node_cnt / mol_node_cnt) * sub1_entropy + (sub2_node_cnt / mol_node_cnt) * sub2_entropy }
        for nodeset in self.dSubgraphSmiles:
            s1, s2 = self.dSubgraphSmiles[nodeset]
            n1, n2 = Chem.MolFromSmarts(s1).GetNumAtoms(), Chem.MolFromSmarts(s2).GetNumAtoms()
            nIG = self.molentropy - ( float(n1/mol_n) * self.dEntropy[s1]  + float(n2/mol_n) * self.dEntropy[s2] )
            self.dIG[s1,s2] = nIG
            
    def choose_argmax(self):
        return max(self.dIG, key=self.dIG.get)
    
    def choose_random(self, equal=False, seed=False):
        if seed:
            random.seed(seed)
        if equal:
            # print(random.choice( list(self.dSubgraphSmiles.values()) ))
            try: return random.choice( [(s1, s2) for (s1, s2) in list(self.dSubgraphSmiles.values()) if Chem.MolFromSmarts(s1).GetNumAtoms() == Chem.MolFromSmarts(s2).GetNumAtoms()] ) 
            except: return random.choice( list(self.dSubgraphSmiles.values()) )
        else:
            return random.choice(list(self.dIG.keys()))

        
def GetDivisionTree(mat, SRWobj, seed = 3078, getMainTree=False, getNewCount=False):
    cnt = 0
    maintree = nx.DiGraph()
    dIndividtree = {}
    print(f"There are {mat.shape[1]} subgraph candidates.")
    for i, subgraph in enumerate(mat.columns):
        moltree = nx.DiGraph()
        nNodeNum = 0
        moltree.add_node(nNodeNum, smiles = subgraph, size = Chem.MolFromSmiles(subgraph, sanitize=False).GetNumAtoms())
        leafnodes = [x for x in moltree.nodes() if moltree.out_degree[x] == 0]
        spawns = 1
        while spawns > 0: # build a tree when there is a division point
        # while ((len(moltree.nodes()) < 3) & (spawns > 0)): 
            # initialize the loop
            spawns = 0
            nNodeNum = max([x for x in moltree.nodes()])
            leafnodes = [x for x in moltree.nodes() if moltree.out_degree[x] == 0]
            for nLeaf in leafnodes:
                smi = moltree.nodes[nLeaf]['smiles']
                test = mol_obj(smi) # from graph_cook.mol_obj
                test.GetMolEntropy(SRWobj)
                test.division()
                # enumerate candidate subgraphs
                for sub in test.candidateSmiles:
                    test.cal_entropy(sub, SRWobj)                    
                if Chem.MolFromSmarts(smi).GetNumAtoms() > 1:  # if there is a possibility of division
                    # calculdate information gain for all the subgraph sets
                    test.cal_information_gain()
                    # choose the subgraph set
                    # s1, s2 = test.choose_argmax()
                    # if (test.dIG[s1,s2] > 0) & ((s1 not in train_X.columns) & (s2 not in train_X.columns)):
                    # if (test.dIG[s1,s2] > 0) & ((s1 not in train_X.columns) | (s2 not in train_X.columns)):
                    # if ((s1 not in train_X.columns) | (s2 not in train_X.columns)):
                    # random selection of subgraphs
                    s1, s2 = test.choose_random(equal=True, seed=False)
                    if ((s1 not in mat.columns) | (s2 not in mat.columns)):
                        spawns += 1
                        moltree.add_node(nNodeNum +1, smiles = s1, entropy = test.molentropy, size = Chem.MolFromSmarts(s1).GetNumAtoms())
                        moltree.add_node(nNodeNum +2, smiles = s2, entropy = test.molentropy, size = Chem.MolFromSmarts(s2).GetNumAtoms())
                        moltree.add_edge(nLeaf, nNodeNum + 1)
                        moltree.add_edge(nLeaf, nNodeNum + 2)
                        nIG = test.dIG[s1,s2]
                # no possibility of division
            # End for: smi in leafnodes
        # End While
        if len(moltree.nodes()) > 1:
            dIndividtree[subgraph] = moltree
            maintree = nx.compose(maintree, moltree)
            cnt += 1
    if getMainTree:
        return dIndividtree, maintree
    return dIndividtree

def GetPrunedMatrix(mat, tree, getNewCount=False, dropFeatures=True):
    # Allocate the fractional value to the SRW vector
    # Input: train_X / valid_X / test_X
    # Output: pruned_train_X / pruned_valid_X / pruned_test_X
    # additional arguments = IndividTreeDictionary from train data
    pruned_mat = mat.copy()
    #
    dweightdict = {}
    newcnt = 0
    for i, (smi, moltree) in enumerate(list(tree.items())):
        rootnode  = 0
        leafnodes = [x for x in moltree.nodes() if moltree.out_degree[x] == 0]
        dweightdict[smi] = {}
        for leaf in leafnodes:
            weight = moltree.nodes[leaf]['size'] / moltree.nodes[rootnode]['size']
            leaf_smi = moltree.nodes[leaf]['smiles']
            dweightdict[smi][leaf] = weight
            try:    
                pruned_mat.loc[:, leaf_smi] = pruned_mat.loc[:, leaf_smi] + mat.loc[:,smi]*weight
            except: 
                pruned_mat[leaf_smi] = mat.loc[:,smi]*weight
                newcnt += 1
        if dropFeatures:
            pruned_mat.drop(columns = [smi], inplace=True) 
    if getNewCount:
        return pruned_mat, newcnt
    else:
        return pruned_mat

def get_fp(x):
    return [mols[ind].HasSubstructMatch(Chem.MolFromSmarts(x.name)) for ind in x.index]

def work_func(data):
    return data.apply(lambda x: get_fp(x))
    
def parallel_dataframe(df, func, num_cores):
    groups = df.groupby(np.arange(len(df.index)) % num_cores)
    df_split = [x[1] for x in groups]
    pool = Pool(num_cores)
    fp_df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return fp_df
    
def GetBinaryMat(mat_pruned_input, mat_full_bin, mols):
    # mols = molinfo_df.molobj
    mat_pruned_bin = mat_pruned_input.copy()
    for c_ind, smi in enumerate(mat_pruned_input.columns):
        try:
            mat_pruned_bin[smi] = mat_full_bin[smi]
        except:
            for r_ind in mat_full_bin.index:
                mat_pruned_bin[smi][r_ind] = mols[r_ind].HasSubstructMatch(Chem.MolFromSmarts(smi), useChirality=True)
    return mat_pruned_bin


def GetLeaves(dtree):
    leaves = []
    for i, (smi, moltree) in enumerate(list(dtree.items())):
        leafnodes = [x for x in moltree.nodes() if moltree.out_degree[x] == 0]
        leaf_smi = [moltree.nodes[leaf]['smiles'] for leaf in leafnodes]
        leaves.append(leaf_smi)
    return i, leaves
##########################################

class mol_separation():
    def DivideIntoTwo(self, _mol, seed = False, returnAtoms = False):
        # select list of nodes
        totalatoms = [x.GetIdx() for x in _mol.GetAtoms()]
        if seed:
            s1, s2 = seed
        else:
            s1, s2 = random.choice(totalatoms, size=2)
        s1, s2 = int(s1), int(s2)
        print(f"Input Graph size: {len(totalatoms)}")
        print(f"Seed Atoms selected: {s1}, {s2}")
        lAtoms_1, lAtoms_2 = [s1], [s2]
        lRemains = list(set(totalatoms) - set(lAtoms_1) - set(lAtoms_2))
        lNeigh_1 = [x.GetIdx() for x in _mol.GetAtomWithIdx(s1).GetNeighbors()]
        lNeigh_2 = [x.GetIdx() for x in _mol.GetAtomWithIdx(s2).GetNeighbors()]
        #print(len(lRemains), len(lNeigh_1), len(lNeigh_2), lAtoms_1, lAtoms_2)
        while len(lRemains) > 0:
            try:
                lNeigh_1 += [x.GetIdx() for x in _mol.GetAtomWithIdx(s1).GetNeighbors()]
                lNeigh_2 += [x.GetIdx() for x in _mol.GetAtomWithIdx(s2).GetNeighbors()]
                lNeigh_1 = list(set(lRemains) & set(lNeigh_1) - set(lAtoms_2))
                lNeigh_2 = list(set(lRemains) & set(lNeigh_2) - set(lAtoms_1))
                next_s1 = int(random.choice(lNeigh_1))
                next_s2 = int(random.choice(lNeigh_2))
                lAtoms_1.append(next_s1)
                lAtoms_2.append(next_s2)
                #
                lRemains = list(set(lRemains) - set(lAtoms_1) - set(lAtoms_2))
                #print(len(lRemains), len(lNeigh_1), len(lNeigh_2), lAtoms_1, lAtoms_2, next_s1, next_s2)
                s1, s2 = next_s1, next_s2
            except:
                if len(lNeigh_1) == 0:
                    lAtoms_2 = list(set(lAtoms_2).union(set(lRemains)))
                elif len(lNeigh_2) == 0:
                    lAtoms_1 = list(set(lAtoms_1).union(set(lRemains)))
                else:
                    print(f"Warning: Selection of nodes given {seed} early stopped with {len(lAtoms_1)} and {len(lAtoms_2)} nodes selected.")
                lRemains = []
        print(f'The size of first/second graphs is {len(lAtoms_1)} / {len(lAtoms_2)}')
        first_graph = Chem.MolFragmentToSmiles( _mol, atomsToUse = lAtoms_1)
        second_graph = Chem.MolFragmentToSmiles( _mol, atomsToUse = lAtoms_2)
        if (Chem.MolFromSmiles(first_graph) is None):
            print("Warning: The first subgraph is not a valid molecule given the generated subgraph.")
        if (Chem.MolFromSmiles(second_graph) is None):
            print("Warning: The second subgraph is not a valid molecule given the generated subgraph.")
        if returnAtoms:
            return first_graph, second_graph, lAtoms_1, lAtoms_2
        else:
            return first_graph, second_graph

    def DivideIntoTwo_random(self, _mol, _fraction = 0.5, _margin = 0.1, nBreakBonds = 1, selected = False, returnAtoms = False):
        from numpy import random
        totalbonds = [[x.GetBeginAtomIdx(), x.GetEndAtomIdx()] for x in _mol.GetBonds()]
        BondObj    = [_mol.GetBondBetweenAtoms(start_idx, end_idx) for start_idx, end_idx in totalbonds ]
        startends  = {x.GetIdx() : [x.GetBeginAtomIdx(), x.GetEndAtomIdx()] for x in _mol.GetBonds()}
        dFragments = {}
        dbondatomdict = {}
        dbondfragdict = {}
        if selected == False: # list
            if type(nBreakBonds) == int:
                if nBreakBonds == 1:
                    for start_idx, end_idx in totalbonds:
                        bondidx = _mol.GetBondBetweenAtoms(start_idx, end_idx)
                        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(_mol, [bondidx.GetIdx()], addDummies=False))
                        if len(frags) == 2:
                            dFragments[bondidx.GetIdx()] = frags
                            dbondfragdict[bondidx.GetIdx()] = frags
                            dbondatomdict[bondidx.GetIdx()] = [start_idx, end_idx]
                else:
                    from itertools import combinations
                    lbondsToBreak = combinations(BondObj, nBreakBonds)
                    #
                    for bonds in lbondsToBreak:
                        bondIDs = [x.GetIdx() for x in bonds]
                        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(_mol, bondIDs, addDummies=False))
                        bondkey = tuple(bondIDs)
                        if len(frags) == 2:
                            dFragments[bondkey] = frags
                            dbondfragdict[bondkey] = frags
                            dbondatomdict[bondkey] = [startends[x] for x in bondIDs]
            #
            print(f"There are {len(dFragments)} candidates.")
            frag_cals = [[key, abs(len(val[0])/(len(val[0]) + len(val[1])) - _fraction)] for key, val in dFragments.items()]
            frag_cands = [key for key, val in frag_cals if abs(val - min([k[1] for k in frag_cals])) < 0.1 ]
            frag_idx = frag_cands[int(np.random.choice(len(frag_cands),1))]
            if nBreakBonds == 1:
                print(f"Broken bond ID: {frag_idx}")
                print(f"Begin/End Atoms of the bond: {dbondatomdict[frag_idx][0]}/{dbondatomdict[frag_idx][1]}")
            else:
                print(f"Broken bond IDs: {list(frag_idx)}")
                print(f"[Begin,End] Atoms of the bond: {dbondatomdict[tuple(frag_idx)]}")
            #print(dFragments[frag_idx])
            first_atomlist, second_atomlist = dFragments[frag_idx]
        else:
            bond1_start_idx, bond1_end_idx = selected[0]
            bond2_start_idx, bond2_end_idx = selected[1]
            bond1_idx = _mol.GetBondBetweenAtoms(bond1_start_idx, bond1_end_idx)
            bond2_idx = _mol.GetBondBetweenAtoms(bond2_start_idx, bond2_end_idx)
            frags = Chem.GetMolFrags(Chem.FragmentOnBonds(_mol, [bond1_idx.GetIdx(), bond2_idx.GetIdx()], addDummies=False))
            bondkey = tuple((bond1_idx.GetIdx(), bond2_idx.GetIdx()))
            if len(frags) == 2:
                dFragments[bondkey] = frags
                dbondfragdict[bondkey] = frags
                dbondatomdict[bondkey] = selected
                print(dFragments[bondkey])
                first_atomlist, second_atomlist = dFragments[bondkey]
            else:
                first_atomlist, second_atomlist = '', ''
                raise Exception(f"There are {len(frags)} fragments produced given {bondkey}.")
        # return two graphs
        first_graph = Chem.MolFragmentToSmiles( _mol, atomsToUse = first_atomlist)
        second_graph = Chem.MolFragmentToSmiles( _mol, atomsToUse = second_atomlist)
        if returnAtoms:
            return [first_graph, second_graph], dbondfragdict, list(first_atomlist), list(second_atomlist)
        else:
            return [first_graph, second_graph], dbondfragdict
        
def subgraph_count_distribution(_mol, subgraph_cnt, _nIter = 0):
    G = mol2graph(_mol)
    vk = dict(subgraph_cnt[_nIter])
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues, Pk

def shannon_entropy_subgraph(G, subgraph_cnt, _nIter = 0):
    import math
    k,Pk = subgraph_count_distribution(G, subgraph_cnt, _nIter)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H


############# Entropy based mol separation ################

class molentropy():
    def __init__(self, SRWobj):
        self.SRWobj = SRWobj
        self.mol = ''
        self.graph = ''
        self.smiles = ''
        self.mol_cnt = ''
        # for entropy
        self.probs = ''
        self.kvalues = ''
        self.entropy = 0.0
    
    def GetMol(self, mol_id=False, smi=False):
        if mol_id:
            self.mol = self.SRWobj.train_molinfo_df.loc[mol_id,'molobj']
            self.smiles = Chem.MolToSmiles(self.mol)
        elif smi:
            self.mol = Chem.MolFromSmiles(smi)
            self.smiles = smi
     
    def mol2graph(self):
        if self.mol == None:
            raise Exception(f"You have provide an empty rdkit.mol object:{self.mol}.")
        atoms_info = [ (atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol()) for atom in self.mol.GetAtoms() ]
        bonds_info = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType(), bond.GetBondTypeAsDouble()) for bond in self.mol.GetBonds()]
        
        self.graph = nx.Graph()
        for atom_info in atoms_info:
            self.graph.add_node(atom_info[0], AtomicNum=atom_info[1], AtomicSymbole=atom_info[2])
        for bond_info in bonds_info:
            self.graph.add_edge(bond_info[0], bond_info[1], BondType=bond_info[2], BondTypeAsDouble=bond_info[3])
   
    def GetSubgraphCount(self, atoms = False):
        self.mol_cnt = pd.DataFrame.from_dict({(nIteration) : self.SRWobj.dNodeSubgraphCount[nIteration][idmax]
                                   for nIteration in range(20)}).sort_index()
        self.mol_cnt.index = self.mol_cnt.index + 1
        if atoms:
            self.mol_cnt = self.mol_cnt.iloc[atoms, :]
    
    def degree_distribution(self):
        vk = dict(self.graph.degree())
        vk = list(vk.values()) # we get only the degree values
        maxk = np.max(vk)
        mink = np.min(min)
        self.kvalues= np.arange(0,maxk+1) # possible values of k
        Pk = np.zeros(maxk+1) # P(k)
        for k in vk:
            Pk[k] = Pk[k] + 1
        self.probs = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one

    def subgraph_count_distribution(self, _nIter = 0):
        vk = dict(self.mol_cnt[_nIter])
        vk = list(vk.values()) # we get only the degree values
        maxk = np.max(vk)
        mink = np.min(min)
        self.kvalues= np.arange(0,maxk+1) # possible values of k
        Pk = np.zeros(maxk+1) # P(k)
        for k in vk:
            Pk[k] = Pk[k] + 1
        self.probs = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
        
    def shannon_entropy(self, measure = 'degree', _nIter = 0):
        self.entropy = 0.0
        import math
        if measure == 'degree':
            self.degree_distribution()
        elif measure == 'prob':
            self.subgraph_count_distribution(_nIter)
        for p in self.probs:
            if(p > 0):
                self.entropy = self.entropy - p*math.log(p, 2)
        return self.entropy
    
    
    
def normalized_laplacian(adj_matrix):
    nodes_degree = np.sum(adj_matrix, axis=1)
    nodes_degree_sqrt = 1/np.sqrt(nodes_degree)
    degree_matrix = np.diag(nodes_degree_sqrt)
    eye_matrix = np.eye(adj_matrix.shape[0])
    return eye_matrix - degree_matrix * adj_matrix * degree_matrix


def unnormalized_laplacian(adj_matrix):
    nodes_degree = np.sum(adj_matrix, axis=1)
    degree_matrix = np.diag(nodes_degree)
    return degree_matrix - adj_matrix

def VNGE_exact(adj_matrix):
    start = time.time()
    nodes_degree = np.sum(adj_matrix, axis=1)
    c = 1.0 / np.sum(nodes_degree)
    laplacian_matrix = c * unnormalized_laplacian(adj_matrix)
    eigenvalues, _ = np.linalg.eig(laplacian_matrix)
    eigenvalues[eigenvalues < 0] = 0
    pos = eigenvalues > 0
    H_vn = - np.sum(eigenvalues[pos] * np.log2(eigenvalues[pos]))
    print('H_vn exact:', H_vn)
    return H_vn
    # print('Time:', time.time() - start)

def VNGE_FINGER(adj_matrix):
    start = time.time()
    nodes_degree = np.sum(adj_matrix, axis=1)
    c = 1.0 / np.sum(nodes_degree)
    edge_weights = 1.0 * adj_matrix[np.nonzero(adj_matrix)]
    approx = 1.0 - np.square(c) * (np.sum(np.square(nodes_degree)) + np.sum(np.square(edge_weights)))
    laplacian_matrix = unnormalized_laplacian(adj_matrix)
    '''
    eigenvalues, _ = np.linalg.eig(laplacian_matrix)  # the biggest reduction
    eig_max = c * max(eigenvalues)
    '''
    eig_max, _ = eigsh(laplacian_matrix, 1, which='LM')
    eig_max = eig_max[0] * c
    H_vn = - approx * np.log2(eig_max)
    print('H_vn approx:', H_vn)
    # print('Time:', time.time() - start)
    

# From Wonseok    
# Refine features using wildcard atoms
import re
atom_groups = [[5, 7], [8, 16], [9, 17, 35, 53]]
replacements = {}
for g in atom_groups:
    for a in g:
        replacements[f"#{a}"] = ','.join(f"#{a}" for a in g) 
        # To use wildcard, replace above with replacements[f"#{a}"] = '*'
rep = dict((re.escape(k), v) for k, v in replacements.items()) 
pattern = re.compile("|".join(rep.keys()))

def compute_entropy_score(train_df, fragment):
    train_matches = [m.HasSubstructMatch(fragment) for m in train_df['molecule']]
    train_labels = train_df["DILI_label"].to_list()
    positive_support, negative_support = 0, 0
    for i in range(len(train_df)):
        if train_matches[i] == 1:
            if train_labels[i] == 1: positive_support += 1
            else: negative_support += 1
    positive_support = positive_support / train_labels.count(1)
    negative_support = negative_support / train_labels.count(0)
    if positive_support == 0 and negative_support == 0: return 0
    if positive_support == 0 or negative_support == 0: return 1
    total_support = positive_support + negative_support
    positive_support /= total_support
    negative_support /= total_support
    return 1 + (positive_support * np.log2(positive_support) + negative_support * np.log2(negative_support))
    
def supervised_relaxation(train_df, smart):
    s_candidate = [smart]
    for it in pattern.finditer(smart):
        s_candidate.append(smart[:it.start()] + replacements[it.group(0)] + smart[it.end():])
    f_candidate = [Chem.MolFromSmarts(s) for s in s_candidate]
    ent_score = [compute_entropy_score(train_df, f) for f in f_candidate]
    relaxed = s_candidate[np.argmax(ent_score)]
    print(f"ORIGINAL {smart:40s} {ent_score[0]:.3f}              RELAXED {relaxed:40s} {np.max(ent_score):.3f}")
    return relaxed

# for i, s in enumerate(fragment_smarts):
#     relaxed_smarts.append(supervised_relaxation(train_df, s))
