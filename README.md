# SSM
Supervised Subgraph Mining for Chemical Graphs

![figure_overview](https://user-images.githubusercontent.com/25650482/201132237-31a1bc7b-9292-479e-91b1-385f6368a31d.JPG)

## Installation
Install with `conda`.
```
conda env create -f environment.yaml
```

## Usage

### Training and Test
```
python bin/ssm_smiles.py --train_data <train.tsv> --test_data <test.tsv> \
            --output_dir <dir> \
            --rw <rw> --alpha <alpha> \
            --iterations <iterations> --nWalker <nWalker> \
            --seed <seed>
```

### Test with a trained model
```
python bin/ssm_smiles.py --test_data <test.tsv> \
            --output_dir <dir> --trained_file model/ssm_trained.pickle \
            --rw <rw> --alpha <alpha> \
            --iterations <iterations> --nWalker <nWalker> \
            --seed <seed>
```

*Options*

- `--train_data`: A tsv file for Training data. "SMILES" and "label" must be included in the header. If you do not provide `train_data` to train the model, DILIstfrom (Chem Res Toxicol, 2021) will be used as training data.
- `--test_data`: A tsv file for Test data. "SMILES" must be included in the header. If the header contains "label", the performance of the model is measured.
- `--outpur_dir`: Path for output directory.
- `--trained_file`: A pickle file (`ssm_trained.pickle`) resulting from training the model.
- `--rw, -l`: Length of random walks. [default: 7]
- `--alpha, -a`: Rate of updating graph transitions. [default: 0.1]
- `--iterations, -k`: Number of iterations. [default: 20]
- `--nWalker`: Number of subgraphs for the augmentation. [default: 5]
- `--seed`: Seed number for reproducibility.

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
