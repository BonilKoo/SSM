# Read Python libraries
import os
import argparse, pickle
import pandas as pd
# from collections import defaultdict

# Read base class
from mychem import *
from mydata import PrepareData
from SSM_main import DILInew, prediction
from utils import *

class get_subgraph():
    def __init__(self, args, chemistry='atom', pruning='pure', rule='random'):
        self.trained = '' # model
        self.test = '' # model
        self.rw = args.rw
        self.chemistry = chemistry
        self.alpha = args.alpha
        self.iterations = args.iterations
        self.pruning = pruning
        self.nWalker = args.nWalker
        self.sRule = rule
        self.nSeed = args.seed
        #
        self.train_df         = pd.DataFrame()
        self.train_molinfo_df = pd.DataFrame()
        self.test_df          = pd.DataFrame()
        self.test_molinfo_df  = pd.DataFrame()

    def read_model(self):
        pretrained_file = open("../model/ssm_trained.pickle", "rb")
        # argmax iter = 1
        self.trained = pickle.load(pretrained_file)
        print("Model loaded\n")

    def read_data(self, data):
        my_data = PrepareData()
        my_data.read_data(test_fname=data)
        self.train_df = my_data.train_df
        self.test_df = my_data.test_df
        self.train_df, self.train_molinfo_df = my_data.prepare_rw_train(self.train_df)
        self.test_df, self.test_molinfo_df = my_data.prepare_rw(self.test_df)
        print("Data Preraparation Complete\n")

# Read data
def SSM_parser():
    parser = argparse.ArgumentParser(description = 'Supervised Subgraph Mining')
    parser.add_argument("--test_data", required=True, help="SMILES data (See test/short_test.tsv)", type=str)
    parser.add_argument("--output_dir", help="Path for output directory. (Default directory: ../results)", default='../../results')

    parser.add_argument('--rw', default=10, type=int, help='Length of random walks.')
    parser.add_argument('--alpha', default=0.9, type=float_range, help='Rate of updating graph transitions.')
    parser.add_argument('--iterations', default=1, type=int, help='Number of iterations.')
    parser.add_argument('--nWalker', default=5, type=int, help='Number of subgraphs for the augmentation.')
    parser.add_argument('--seed', default=0, type=int, help='Seed number for reproducibility.')

    args = parser.parse_args()
    return args

def main():
    print(f"Current working directory: {os.getcwd()}\n")
    # Initialize and read trained model
    args = SSM_parser()
    seed_everything(args.seed)
    # Load main SSM class
    ssm = get_subgraph(args)
    ssm.read_data(args.test_data)
    ssm.read_model()
    # Run Supervised Subgraph Mining for the test data
    ssm.test = DILInew(chemistry = ssm.chemistry, n_rw = ssm.rw, n_alpha = ssm.alpha, iteration = ssm.iterations, pruning = ssm.pruning, n_walker = ssm.nWalker , rw_mode = ssm.sRule)
    ssm.test.valid(ssm.test_molinfo_df, ssm.train_molinfo_df, ssm.trained.dEdgeClassDict, ssm.trained.dFragSearch)
    os.makedirs(args.output_dir, exist_ok=True)
    valid_archive = open(f'{args.output_dir}/ssm.pickle', 'wb')
    pickle.dump(ssm.test, valid_archive, pickle.HIGHEST_PROTOCOL)
    prediction(ssm.trained, ssm.test, ssm.iterations, args.output_dir, ssm.train_molinfo_df, ssm.test_molinfo_df, ssm.nSeed)

if __name__ == '__main__':
    main()
