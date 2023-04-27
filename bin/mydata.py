import os
from collections import defaultdict
import pandas as pd  
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from pysmiles import read_smiles
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL) 
# to return smiles, class information

class myobj():
    def __init__(self, sanitize=True):
        self.original_data  = pd.DataFrame() # columns = ['smiles', 'class']
        self.train_df       = pd.DataFrame() # columns = ['smiles', 'class']
        self.test_df        = pd.DataFrame() 
        self.sanitize       = sanitize
    def sanitize_mols(self, row):
        remover = SaltRemover()
        index = row.iloc[0]
        name = row.iloc[1]
        mol = Chem.MolFromSmiles(row["smiles"])
        ChemMol = remover.StripMol(mol)
        try: smiles = Chem.MolToSmiles(ChemMol, isomericSmiles=True, canonical=True, rootedAtAtom=-1)
        except: smiles = Chem.MolToSmiles(ChemMol, isomericSmiles=True, canonical=True)
        row['smiles'] = smiles
        return row
    def read_data(self, test_fname):
        print(f"Training Data: DILIst from (Chem Res Toxicol, 2021)")
        self.original_data = pd.read_csv("../training_data/deepdili_dilist_train_all.tsv", sep="\t", header="infer")
        self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
        try:    self.train_df  = pd.DataFrame(self.original_data[["CID", "Canonical SMILES", "DILI_label"]])  # for others
        except: self.train_df  = pd.DataFrame(self.original_data[["Chemical name", "Canonical SMILES", "DILI_label"]])  # for Liew data
        self.train_df.columns = ["CID", "smiles", "class"] # columns of training data
        # read test
        print(f"Read Test Data: {test_fname}")
        self.test_df = pd.read_csv(test_fname, sep="\t", header="infer") # ['index', 'name', 'smiles']
        self.test_df.columns    = ['Number', 'Compound_Name', 'smiles'] # columns of test data
        def run_sanitize(df):
            if self.sanitize:
                df   = df.apply(self.sanitize_mols, axis=1)
            return df
        self.train_df = run_sanitize(self.train_df)
        self.test_df  = run_sanitize(self.test_df)
        self.train_df.reset_index(inplace=True, drop=True)
        self.test_df.reset_index(inplace=True, drop=True)
        return self.train_df, self.test_df
   
class PrepareData():
    def __init__(self, sanitize=True, valid_external=False, mode = "cv"):
        self.mydata          = myobj(sanitize=sanitize)
        self.mode            = mode
        self.train_df, self.test_df = defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)
    def read_data(self, test_fname):
        self.train_df, self.test_df = self.mydata.read_data(test_fname = test_fname)
        print(f'Training Data shape: {self.train_df.shape}')
        print(f'Test Data shape: {self.test_df.shape}\n')
    def prepare_rw(self, data): # remove chemicals without edges
        molinfo_df = pd.DataFrame( columns = ["Number", "smiles", "molobj", "molgraph"])
        removed    = []
        for ind in data.index: # iterate over molecules
            smiles = data['smiles'][ind]
            rdMol = Chem.MolFromSmiles(smiles) # rdkit.Chem.rdchem.Mol object
            nxMol = read_smiles(smiles, reinterpret_aromatic=True) # networkx.classes.graph.Graph
            if ('.' in smiles):
                removed.append(ind)
            else:
                if nxMol.number_of_nodes() > 1:
                    molinfo_df = molinfo_df.append( dict(zip(molinfo_df.columns.to_list(), 
                                                                [ind, smiles, rdMol, nxMol ])), 
                                                                ignore_index=True )
                else:
                    removed.append(diliid)
        #molinfo_df.reset_index(drop=True, inplace=True)
        molinfo_df.index = molinfo_df.Number
        if len(removed) != 0:
            print(f'Molecules not allowed to Random Walks in Test Data:')
            for i in removed:
                smiles = data['smiles'][i]
                print(f'\t{i}\t{smiles}')
        res_data = data[~data.index.isin(removed)]
        print(f'The shape of RW-allowed DILI molecules in Test Data: {res_data.shape}\n')
        return res_data, molinfo_df

    def prepare_rw_train(self, data): # remove chemicals without edges
        molinfo_df = pd.DataFrame( columns = ['ID','class','smiles','molobj','molgraph'] )
        removed    = []
        for diliid in data.index: # iterate over molecules
            smiles = data['smiles'][diliid]
            rdMol = Chem.MolFromSmiles(smiles) # rdkit.Chem.rdchem.Mol object
            nxMol = read_smiles(smiles, reinterpret_aromatic=True) # networkx.classes.graph.Graph
            if ('.' in smiles):
                removed.append(diliid)
            else:
                if nxMol.number_of_nodes() > 1:
                    molinfo_df = molinfo_df.append( dict(zip(molinfo_df.columns.to_list(), 
                                                                [diliid, data['class'][diliid], smiles, rdMol, nxMol ])), 
                                                                ignore_index=True )
                else:
                    removed.append(diliid)
        #molinfo_df.reset_index(drop=True, inplace=True)
        molinfo_df.index = molinfo_df.ID
        if len(removed) != 0:
            print(f'Molecules not allowed to Random Walks in Training Data:')
            for i in removed:
                smiles = data['smiles'][i]
                print(f'\t{i}\t{smiles}')
        res_data = data[~data.index.isin(removed)]
        print(f'The shape of RW-allowed DILI molecules in Training Data: {res_data.shape}\n')
        return res_data, molinfo_df
