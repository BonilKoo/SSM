from collections import defaultdict
import pandas as pd  
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from pysmiles import read_smiles
# to return smiles, class information

class myobj():
    def __init__(self, sanitize=True):
        self.original_data  = pd.DataFrame() # columns = ['smiles', 'class']
        self.df             = pd.DataFrame() # columns = ['smiles', 'class']
        self.sanitize       = sanitize
        self.source         = ''
        self.dDatabase      = {"jcim_comb_train":"Combined_training", "jcim_liew_train":"Liew_training" }
    def sanitize_mols(self, row):
        remover = SaltRemover()
        name = row.iloc[0]
        mol = Chem.MolFromSmiles(row["smiles"])
        ChemMol = remover.StripMol(mol)
        try: smiles = Chem.MolToSmiles(ChemMol, isomericSmiles=True, canonical=True, rootedAtAtom=-1)
        except: smiles = Chem.MolToSmiles(ChemMol, isomericSmiles=True, canonical=True)
        return pd.Series([name, smiles, row["class"]])
    def read_data(self, name="comb_train"):
        print(f"Read Data: {name}")
        self.df       = pd.DataFrame()
        if "jcim" in name:
            self.source = 'jcim'
            if name == "jcim_comb_train":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_Combined_training.tsv", sep="\t", header="infer")
            elif name == "jcim_comb_valid":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_Combined_validation.tsv", sep="\t", header="infer")
            elif name == "jcim_nctr_train":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_NCTR_training.tsv", sep="\t", header="infer")
            elif name == "jcim_nctr_valid":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_NCTR_validation.tsv", sep="\t", header="infer")
            elif name == "jcim_greene_valid":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_Greene_validation.tsv", sep="\t", header="infer")
            elif name == "jcim_xu_valid":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_Xu_validation.tsv", sep="\t", header="infer")
            elif name == "jcim_liew_train":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_Liew_training.tsv", sep="\t", header="infer")
            elif name == "jcim_liew_valid":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/JCIM_2015_Liew_validation.tsv", sep="\t", header="infer")
            self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
            try:    self.df       = pd.DataFrame(self.original_data[["CID", "Canonical SMILES", "Label"]])  # for others
            except: self.df       = pd.DataFrame(self.original_data[["Chemical name", "Canonical SMILES", "Label"]])  # for Liew data
            self.df.columns = ["CID", "smiles", "class"]
        elif "dili" in name:
            self.source = 'fda'
            if name == "dilist":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/dilist_2020_smiles.tsv", sep="\t", header="infer")
                self.original_data = self.original_data[~self.original_data["smiles"].isna()]
                self.df       = pd.DataFrame(self.original_data[["Compound_Name", "smiles", "class"]])
            elif name == "dilirank": # of only most & no
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/DILIrank-List.csv", sep=",", header="infer")
                self.original_data = self.original_data[~self.original_data["SMILES"].isna()]
                self.df       = pd.DataFrame(self.original_data[["Compound_Name", "SMILES", "vDILIConcern"]])
                self.df       = self.df.loc[(self.df["vDILIConcern"]=="vNo-DILI-Concern") | (self.df["vDILIConcern"]=="vMost-DILI-Concern")]
                self.df["vDILIConcern"].replace([1,3], [0,1], inplace=True)
            elif name == "dilirank_ex": # of only most & no
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/DILIrank-List.csv", sep=",", header="infer")
                self.original_data = self.original_data[~self.original_data["SMILES"].isna()]
                self.df       = pd.DataFrame(self.original_data[["Compound_Name", "SMILES", "vDILIConcern"]])
                self.df       = self.df.loc[(self.df["vDILIConcern"]=="vNo-DILI-Concern") | (self.df["vDILIConcern"]=="vLess-DILI-Concern") | (self.df["vDILIConcern"]=="vMost-DILI-Concern")]
                self.df["vDILIConcern"].replace([1,2,3], [0,1,1], inplace=True)
            elif name == "dilirank_all": # of only most & no
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/DILIrank-List.csv", sep=",", header="infer")
                self.original_data = self.original_data[~self.original_data["SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["SMILES"]!="."]
                self.df       = pd.DataFrame(self.original_data[["Compound_Name", "SMILES", "vDILIConcern"]])
                self.df["vDILIConcern"].replace(["Ambiguous DILI-concern","vNo-DILI-Concern","vLess-DILI-Concern","vMost-DILI-Concern"], [0,1,2,3], inplace=True)
                print("Warning: Class labels should be re-defined before applying to models (amb:0, No:1, Less:2, Most:3)")
            elif name == "dilist_deepdili_train_all":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_dilist_train_all.tsv", sep="\t", header="infer") # n = 753
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
            elif name == "dilist_deepdili_train":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_dilist_train.tsv", sep="\t", header="infer") # n = 602
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
            elif name == "dilist_deepdili_valid":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_dilist_valid.tsv", sep="\t", header="infer") # n = 151
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
            elif name == "dilist_deepdili_test": # from deepdili
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_dilist_test.tsv", sep="\t", header="infer") # n = 249
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
            elif name == "deepdili_nctr": # from deepdili
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_nctr_test.tsv", sep="\t", header="infer") # n = 249
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
            elif name == "deepdili_greene": # from deepdili
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_greene_test.tsv", sep="\t", header="infer") # n = 249
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
            elif name == "deepdili_xu": # from deepdili
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/deepdili_xu_test.tsv", sep="\t", header="infer") # n = 249
                self.original_data = self.original_data[~self.original_data["Canonical SMILES"].isna()]
                self.original_data = self.original_data[self.original_data["Canonical SMILES"]!="."]
                self.df            = pd.DataFrame(self.original_data[["CompoundName", "Canonical SMILES", "DILI_label"]])
                
            elif name == "dilirank_amb":
                self.original_data = pd.read_csv("/data/project/sslim/drug/DILI/data/other_hepatotoxicity/DILIrank_ambiguous_to_evaluate.tsv", sep="\t", header="infer")
                self.original_data = self.original_data[~self.original_data["SMILES"].isna()]
                self.df       = pd.DataFrame(self.original_data[["Compound_Name","SMILES", "FDA_label"]]) # <- caution here
                self.df       = self.df.loc[(self.df["FDA_label"]=="vNo-DILI-Concern") | (self.df["FDA_label"]=="vLess-DILI-Concern") | (self.df["FDA_label"]=="vMost-DILI-Concern")]
                self.df["FDA_label"].replace(["vNo-DILI-Concern","vLess-DILI-Concern","vMost-DILI-Concern"], [0,1,1], inplace=True)
            self.df.columns = ['Compound_Name', 'smiles', 'class']
            #self.original_data.loc[:,"Compound_Name"] = self.original_data["Compound_Name"].apply(lambda x: x.capitalize())
        if self.sanitize:
            self.df   = self.df.apply(self.sanitize_mols, axis=1)
            self.df.columns = ['Compound_Name', 'smiles', 'class']
        self.original_data.reset_index(inplace=True, drop=True)
        self.df.reset_index(inplace=True, drop=True)
        return self.original_data, self.df
    
class PrepareData():
    def __init__(self, sanitize=True, valid_external=False, mode = "cv"):
        self.mydata          = myobj(sanitize=sanitize)
        self.mode            = mode
        self.org_train_df, self.org_valid_df = defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)
        self.train_df, self.valid_df         = defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)
    def read_data(self, train="dilist", valid=False):
        if valid == "single":
            self.mode = "independent"
            self.org_train_df, self.train_df = self.mydata.read_data(name=train)
            print(f'Independent Data shape: {self.train_df.shape}')
        elif valid:
            self.mode = "validation"
            self.org_train_df, self.train_df = self.mydata.read_data(name=train)
            self.org_valid_df, self.valid_df = self.mydata.read_data(name=valid)
            # removal of duplicates from validation data
            print(f'Training Data shape: {self.train_df.shape}')
            print(f'Validation Data shape: {self.valid_df.shape}')
        else:
            print("Internal Cross-validation mode ON")
            from sklearn.model_selection import KFold
            org_df, my_df  = self.mydata.read_data(name=train)
            kf = KFold(n_splits=5, shuffle=True, random_state=3078)
            for idx, (train_index, valid_index) in enumerate(kf.split(my_df)):
                self.org_train_df[idx], self.org_valid_df[idx] = org_df.iloc[org_df.index.isin(train_index),:], org_df.iloc[org_df.index.isin(valid_index),:]
                self.train_df[idx], self.valid_df[idx]         = my_df.iloc[my_df.index.isin(train_index),:], my_df.iloc[my_df.index.isin(valid_index),:]
                #self.train_df[idx].reset_index(inplace=True)
                #self.valid_df[idx].reset_index(inplace=True)
                self.train_df[idx].columns = ["Name", "smiles", "class"]
                self.valid_df[idx].columns = ["Name", "smiles", "class"]
            print(f'Training Data shape: {[self.train_df[x].shape for x in range(5)]}')
            print(f'Validation Data shape: {[self.valid_df[x].shape for x in range(5)]}')
    def prepare_rw(self, data): # remove chemicals without edges
        molinfo_df = pd.DataFrame( columns = ['ID','class','smiles','molobj','molgraph'] )
        removed    = []
        for diliid in data.index: # iterate over molecules
            smiles = data['smiles'][diliid]
            rdMol = Chem.MolFromSmiles(smiles)
            nxMol = read_smiles(smiles, reinterpret_aromatic=True)
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
        print(f'Molecules with no Random Walks allowed: {removed}')
        res_data = data[~data.index.isin(removed)]
        print(f'The shape of RW-allowed DILI molecules {res_data.shape}')
        return res_data, molinfo_df
