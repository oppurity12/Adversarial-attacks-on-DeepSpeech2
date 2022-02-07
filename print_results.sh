for i in $(seq 1 6)
do
 echo epochs75_2conv_$i.ckpt
 python test.py model.model_path=/home/skku/ML/deepspeech.pytorch/checkpoints/epochs75_2conv_$i.ckpt test_path=/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json
done

for i in $(seq 1 6)
do
 echo epochs75_2conv_shortcut_$i.ckpt
 python test.py model.model_path=/home/skku/ML/deepspeech.pytorch/checkpoints/epochs75_2conv_shortcut_$i.ckpt shortcut_cfg=True test_path=/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json
done

for i in $(seq 1 6)
do
 echo epochs75_2conv_uni_$i.ckpt
 python test.py model.model_path=/home/skku/ML/deepspeech.pytorch/checkpoints/epochs75_2conv_uni_$i.ckpt test_path=/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json
done

for i in $(seq 1 6)
do
 echo epochs75_2conv_uni_shortcut_$i.ckpt
 python test.py model.model_path=/home/skku/ML/deepspeech.pytorch/checkpoints/epochs75_2conv_uni_shortcut_$i.ckpt shortcut_cfg=True test_path=/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json
done
