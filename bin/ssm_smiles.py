# Read Python libraries
import os
import argparse, pickle
import pandas as pd
# from collections import defaultdict

# Read base class
from srw_class.set_device_seed import *
from mychem import *
from mydata import PrepareData
from SSM_main import DILInew, prediction

class get_subgraph():
    def __init__(self, rw=10, chemistry='atom', nAlpha=0.9, iterations=1, pruning='pure', nWalker=5, rule='random', mySeed = 0):
        self.trained = '' # model
        self.test = '' # model
        self.rw = rw
        self.chemistry = chemistry
        self.alpha = nAlpha
        self.iterations = iterations
        self.pruning = pruning
        self.nWalker = nWalker
        self.sRule = rule
        self.nSeed = mySeed
        #
        self.train_df         = pd.DataFrame()
        self.train_molinfo_df = pd.DataFrame()
        self.test_df          = pd.DataFrame()
        self.test_molinfo_df  = pd.DataFrame()

    def read_model(self):
        pretrained_file = open("../model/dilist_dili_train_Method2LRTrainEntropyUpdate_graph_rw7_alpha1_Pathrandom_PruneFalse_walkers5_iteration20.pickle", "rb")
        # argmax iter = 1
        self.trained = pickle.load(pretrained_file)
        print("Model loaded")
        
    def read_data(self, data, myPath = '.'):
        my_data = PrepareData()
        my_data.read_data(test_fname=data, myPath = myPath)
        self.train_df = my_data.train_df
        self.test_df = my_data.test_df
        self.train_df, self.train_molinfo_df = my_data.prepare_rw_train(self.train_df)
        self.test_df, self.test_molinfo_df = my_data.prepare_rw(self.test_df)
        print("Data Preraparation Complete")

# Read data
def SSM_parser():
    parser = argparse.ArgumentParser(description = 'Supervised Subgraph Mining')
    parser.add_argument("--smiles_data", "-s", required=True, help="SMILES data (See smiles_example.tsv)", type=str)
    parser.add_argument("--output_name", help="output filename. (Default directory: results/)", default='test_res')
    args = parser.parse_args()
    return args

def main():
    mydir = os.getcwd() + "/"
    print(f"Current working directory: {mydir}")
    # Initialize and read trained model
    args = SSM_parser()
    # Load main SSM class
    ssm = get_subgraph()
    ssm.read_data(args.smiles_data, myPath = mydir)
    ssm.read_model()
    # Run Supervised Subgraph Mining for the test data
    ssm.test = DILInew(chemistry = ssm.chemistry, n_rw = ssm.rw, n_alpha = ssm.alpha, iteration = ssm.iterations, pruning = ssm.pruning, n_walker = ssm.nWalker , rw_mode = ssm.sRule)
    ssm.test.valid(ssm.test_molinfo_df, ssm.train_molinfo_df, ssm.trained.dEdgeClassDict, ssm.trained.dFragSearch)
    os.makedirs(mydir + '../results/', exist_ok=True)
    valid_archive = open(mydir + '../results/' + str(args.output_name + ".pickle"), 'wb')
    pickle.dump(ssm.test, valid_archive, pickle.HIGHEST_PROTOCOL)
    prediction(ssm.trained, ssm.test, ssm.iterations, args.output_name, ssm.train_molinfo_df, ssm.test_molinfo_df, args.output_name, mydir, ssm.nSeed)
    
if __name__ == '__main__':
    main()
