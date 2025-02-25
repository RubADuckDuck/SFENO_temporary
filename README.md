# SFENO

## Introduction 
- SFENO represents "**S**ignal **F**low **E**stimation based on **N**eural **O**DE".


## Run Code

### Run training on Synthetic Data
First generate synthesized data.
```
python run_2_gen_rand_net_data.py --network_size 100
```

Run training
```
python run_train_synth_cellbox.py -b 16 ---num_gpu 4 -e 4000 --ddp --network_size 100
or
python run_train_synth_exampleNN.py -b 16 ---num_gpu 4 -e 4000 --ddp --network_size 100
```

### Run training on given data
first put conds.tsv and exp.tsv file under the following path
```
.../sfeno/datasets/(dataset_name)/
```
like below
```
.../sfeno/datasets/(dataset_name)/conds.tsv
.../sfeno/datasets/(data_name)/exp.tsv
```
set current directory to '.../sfeno/dataset' and run script below
```
python data_converter.py
```
This code will change existing data suitable for Training

If the code ran successfully then you would see files like below   
```
.../sfeno/datasets/(dataset_name)/sfeno_data/conds.tsv   
.../sfeno/datasets/(dataset_name)/sfeno_data/expr.tsv   
.../sfeno/datasets/(dataset_name)/sfeno_data/node_Index.json   
```

Run training 
```
python run_train -b 16 ---num_gpu 4 -e 4000 --ddp --data_path .../sfeno/datasets/(dataset_name)/sfeno_data
```

To run on all existing data run
```
python run_on_all_existing_data -b 16 ---num_gpu 4 -e 4000 --ddp
```

predictions of tests would be saved under 
```
.../sfeno/datasets/(dataset_name)/results/(test_index)
```

Check manuscript for ploting