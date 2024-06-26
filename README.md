# SSM
Supervised Subgraph Mining for Chemical Graphs

![figure_overview](https://user-images.githubusercontent.com/25650482/201132237-31a1bc7b-9292-479e-91b1-385f6368a31d.JPG)

## Configuring environment
Install `conda` environment to run SSM.
```
conda env create -f environment.yaml
```

## Usage

### Training and Test
```
python bin/ssm_smiles.py --train_data <train.csv> \
                         --test_data <test.csv> \
                         --output_dir <dir> \
                         --rw <rw> \
                         --alpha <alpha> \
                         --iterations <iterations> \
                         --nWalker <nWalker>
```

### Test with a trained model
```
python bin/ssm_smiles.py --test_data <test.csv> \
                         --output_dir <dir> \
                         --trained_file model/ssm_train.pickle
```

*Options*

- `--train_data`: a csv file for Training data. "SMILES" and "label" must be included in the header. If you do not provide `train_data` to train the model, DILIst from (Chem Res Toxicol, 2021) will be used as training data.
- `--test_data`: a csv file for Test data. "SMILES" must be included in the header. If the header contains "label", the performance of the model is measured.
- `--output_dir`: path for output directory.
- `--trained_file`: a pickle file (`ssm_train.pickle`) resulting from training the model.
- `--rw, -l`: length of random walks. [default: 7]
- `--alpha, -a`: rate of updating graph transitions. [default: 0.1]
- `--iterations, -k`: number of iterations. [default: 20]
- `--nWalker`: number of subgraphs for the augmentation. [default: 5]
- `--seed`: seed number for reproducibility. [default: 0]
- `--DiSC`: mining discriminative subgraph combinations (DiSCs) from subgraph features (condition: `significance > 0.1` & `support > 0.02`).

## Output files
- `ssm_train.pickle`: a pickle file for saving training object.
- `ssm_test.pickle`: a pickle file for saving test object.
- `result.tsv`: a file containing performance of the model on test data.
- `confusion_matrix.tsv`: a file containing confusion matrix for the model on test data.
- `iteration_{n}/predictions.tsv`: a file containing prediction result of test data for each iteration.
- `iteration_{n}/subgraph.tsv`: a file containing support, entropy and feature importance of each subgraph.
- `iteration_{n}/subgraph_important.tsv`: a file containing subgraphs with "entropy < 0.5" and "feature importance > 0.0001" in classification.
- `iteration_{n}/subgraph_SA.tsv`: a file containing subgraphs of greater support in "class 1" than in "class 0" with more than "1% support" among important subgraphs.
- `iteration_{n}/DiSC.tsv`: a file containing DiSCs. If there is no DiSC that satisfies the conditions, no file is created.

## Citation
```
@article{lim2023supervised,
  title={Supervised chemical graph mining improves drug-induced liver injury prediction},
  author={Lim, Sangsoo and Kim, Youngkuk and Gu, Jeonghyeon and Lee, Sunho and Shin, Wonseok and Kim, Sun},
  journal={iScience},
  volume={26},
  number={1},
  year={2023},
  publisher={Elsevier}
}
```
