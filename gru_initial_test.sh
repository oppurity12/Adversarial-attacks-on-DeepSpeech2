#!/bin/bash

for i in $(seq 1 5)
do
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/gru2 checkpoint.filename=epochs75_2conv_bi_$i log_dir_name=epochs75_2conv_bi_$i number_of_layers=2 skip_steps=0 log_dir_path=/home/skku/ML/deepspeech.pytorch/gru2
done


python save.py checkpoint_dir=/home/skku/ML/deepspeech.pytorch/gru2 save_dir=/home/skku/ML/deepspeech.pytorch/gru2_results test_path=/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json
