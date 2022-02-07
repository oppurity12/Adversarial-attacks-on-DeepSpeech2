python train.py +configs=librispeech \
trainer.max_epochs=70 \
checkpoint.dirpath=/home/skku/ML/deepspeech.pytorch/checkpoints \
checkpoint.filename=epochs75_2conv_bi_$i \
log_dir_name=epochs75_2conv_bi_$i \
number_of_layers=2 skip_steps=0 \
log_dir_path=/home/skku/ML/deepspeech.pytorch/checkpoints4