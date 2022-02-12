#!/bin/bash

for i in $(seq 3 5)
do
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_2conv_bi_$i log_dir_name=epochs75_2conv_bi_$i number_of_layers=2 skip_steps=0 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_2conv_bi_skip1_$i log_dir_name=epochs75_2conv_bi_skip1_$i number_of_layers=2 skip_steps=1 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_2conv_bi_skip2_$i log_dir_name=epochs75_2conv_bi_skip2_$i number_of_layers=2 skip_steps=2 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4

 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_2conv_uni_$i log_dir_name=epochs75_2conv_uni_$i number_of_layers=2 skip_steps=0 model=unidirectional log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_2conv_uni_skip1_$i log_dir_name=epochs75_2conv_uni_skip1_$i number_of_layers=2 skip_steps=1 model=unidirectional log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_2conv_uni_skip2_$i log_dir_name=epochs75_2conv_uni_skip2_$i number_of_layers=2 skip_steps=2 model=unidirectional log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4

 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_3conv_bi_$i log_dir_name=epochs75_3conv_bi_$i number_of_layers=3 skip_steps=0 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_3conv_bi_skip1_$i log_dir_name=epochs75_3conv_bi_skip1_$i number_of_layers=3 skip_steps=1 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4

 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_3conv_uni_$i log_dir_name=epochs75_3conv_uni_$i number_of_layers=3 skip_steps=0 model=unidirectional log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_3conv_uni_skip1_$i log_dir_name=epochs75_3conv_uni_skip1_$i number_of_layers=3 skip_steps=1 model=unidirectional log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4

 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_4conv_bi_$i log_dir_name=epochs75_4conv_bi_$i number_of_layers=4 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_4conv_bi_skip1_$i log_dir_name=epochs75_4conv_bi_skip1_$i number_of_layers=4 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4 skip_steps=1
 python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_4conv_bi_skip2_$i log_dir_name=epochs75_4conv_bi_skip2_$i number_of_layers=4 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4 skip_steps=2

done

python save.py checkpoint_dir=/home/skku/ML/deepspeech.pytorch/checkpoints4 save_dir=/home/skku/ML/deepspeech.pytorch/experiment_results4 test_path=/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json
#cp /home/skku/ML/deepspeech.pytorch/experiment_results2/공격결과.ipynb /home/skku/ML/deepspeech.pytorch/experiment_results4

#python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_4conv_bi_1 log_dir_name=epochs75_4conv_bi_1 number_of_layers=4 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4
#python train.py +configs=an4 trainer.max_epochs=75 checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints4 checkpoint.filename=epochs75_4conv_bi_skip2_1 log_dir_name=epochs75_4conv_bi_skip2_1 number_of_layers=4 log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4 skip_steps=2

