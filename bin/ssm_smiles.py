# Read Python libraries
import os
import argparse
import pickle
import pandas as pd

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

    def read_model(self, model_file):
        pretrained_file = open(model_file, "rb")
        # argmax iter = 1
        self.trained = pickle.load(pretrained_file)
        print("\nTrained model loaded\n")

    def read_data(self, train_data, test_data):
        my_data = PrepareData()
        my_data.read_data(train_fname=train_data, test_fname=test_data)
        self.train_df = my_data.train_df
        self.test_df = my_data.test_df
        self.train_df, self.train_molinfo_df = my_data.prepare_rw_train(self.train_df)
        self.test_df, self.test_molinfo_df = my_data.prepare_rw(self.test_df)
        print("Data Preraparation Complete\n")

# Read data
def SSM_parser():
    parser = argparse.ArgumentParser(description = 'Supervised Subgraph Mining')

    parser.add_argument('--train_data', default=None, type=str, help='A tsv file for Training data. "SMILES" and "label" must be in the header. (Default: DILIst from (Chem Res Toxicol, 2021))')
    parser.add_argument('--test_data', required=True, type=str, help='A tsv file for Test data. "SMILES" must be in the header. (See test/short_test.tsv)')
    parser.add_argument('--output_dir', default='../../results', type=str, help="Path for output directory. (Default directory: ../results)")
    parser.add_argument('--pretrained_file', default=None, type=str, help='A pickle file for pre-trained model. (See ../model/ssm_trained.pickle)')

    parser.add_argument('--rw', default=7, type=int, help='Length of random walks.')
    parser.add_argument('--alpha', default=0.5, type=float_range, help='Rate of updating graph transitions.')
    parser.add_argument('--iterations', default=20, type=int, help='Number of iterations.')
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
    ssm.read_data(args.train_data, args.test_data)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run Supervised Subgraph Mining for the training data
    if args.pretrained_file is not None:
        ssm.read_model(args.pretrained_file)
    else:
        ssm.train = DILInew(chemistry = ssm.chemistry, n_rw = ssm.rw, n_alpha = ssm.alpha, iteration = ssm.iterations, pruning = ssm.pruning, n_walker = ssm.nWalker , rw_mode = ssm.sRule)
        ssm.train.train(ssm.train_molinfo_df)
        train_archive = open(f'{args.output_dir}/ssm_train.pickle', 'wb')
        pickle.dump(ssm.train, train_archive, pickle.HIGHEST_PROTOCOL)
        ssm.read_model(f'{args.output_dir}/ssm_train.pickle')
    
    # Run Supervised Subgraph Mining for the test data
    ssm.test = DILInew(chemistry = ssm.chemistry, n_rw = ssm.rw, n_alpha = ssm.alpha, iteration = ssm.iterations, pruning = ssm.pruning, n_walker = ssm.nWalker , rw_mode = ssm.sRule)
    ssm.test.valid(ssm.test_molinfo_df, ssm.train_molinfo_df, ssm.trained.dEdgeClassDict, ssm.trained.dFragSearch)
    valid_archive = open(f'{args.output_dir}/ssm_test.pickle', 'wb')
    pickle.dump(ssm.test, valid_archive, pickle.HIGHEST_PROTOCOL)
    prediction(ssm.trained, ssm.test, ssm.iterations, args.output_dir, ssm.train_molinfo_df, ssm.test_molinfo_df, ssm.nSeed)

if __name__ == '__main__':
    main()
