# BPartIS
**B**ayesian **Part**icle **I**nstance **S**egmentation for Electron Microscopy Image Quantification

## Training Models

If you are interested in training **bpartis** from scratch, or you would like to reproduce our results, follow these steps:

#### Installation

1. Clone the repository
```console
git clone https://github.com/by256/bpartis.git
```

2. Install requirements
```console
python3 -m pip install -r requirements.txt
```

<!-- #### Unsupervised Pre-Training (Optional)

If you are not interested in pre-training on the [SEM dataset](https://www.nature.com/articles/sdata2018172) then skip this section. Otherwise, you first need to download the SEM dataset and preprocess it as follows.

3. Download the data files [here](https://b2share.eudat.eu/records/b9abc4a997f8452aa6de4f4b7335e582) and place the individual category folders into a single directory.

4. Run `preprocess_sem_data.py`, passing as arguments the directory containing the SEM dataset category folders, and the destination you would like the preprocessed data to be saved to:

```console
python preprocess_sem_data.py --cat-dir=<cat_dir_path> --save-dst=<save_path>
```

5. Pre-train the model on the SEM dataset, passing as an argument the directory containing the preprocessed data:

```console
python bpartis/pretrain.py --data-dir=<> -->
```

#### Training BPartIS

6. Download the EMPS dataset from here. UPDATE THIS WHEN DATA LOCATION IS DECIDED.

7. Train the BPartIS model on the EMPS dataset. If you pre-trained on the SEM dataset, be sure to use the --finetune argument.

```console
python bpartis/train.py --data-dir=data/ --finetune=1 --device=cuda --epochs=300 --save-dir=bpartis/saved_models/
```
