from configs.attack_config import AttackConfig
from deepspeech_attack.attacks.attack import run
from glob import glob

import pandas as pd
import re
import os

cfg = AttackConfig()


def get_adv_wer_adv_cer(model_path, test_path, df, epsilon=0.2, attack_type='fgsm'):
    cfg = AttackConfig()
    cfg.model.model_path = model_path
    cfg.test_path = test_path
    cfg.epsilon = epsilon
    cfg.attack_type = attack_type
    wer, cer = run(cfg)
    data = [[model_path, wer, cer]]
    result = pd.DataFrame(data, columns=['model_path', 'adv_wer', 'adv_cer'])
    df = df.append(result)
    return df


test_path = '/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json'
model_lists = list(glob("checkpoints2/*.ckpt"))


df1 = pd.DataFrame()  # 2conv, uni, skip1
df2 = pd.DataFrame()  # 2conv, uni, skip2
df3 = pd.DataFrame()  # 2conv, uni

df4 = pd.DataFrame()  # 2conv, bi skip1
df5 = pd.DataFrame()  # 2conv, bi skip2
df6 = pd.DataFrame()  # 2conv, bi

df7 = pd.DataFrame()  # 3conv, uni, skip1
df8 = pd.DataFrame()  # 3conv, uni

df9 = pd.DataFrame()  # 3conv, bi, skip1
df10 = pd.DataFrame()  # 3conv, bi

p1 = re.compile(r'epochs\d+_2conv_uni_\d+.ckpt')
p2 = re.compile(r'epochs\d+_2conv_uni_skip1_\d+.ckpt')
p3 = re.compile(r'epochs\d+_2conv_uni_skip2_\d+.ckpt')

p4 = re.compile(r'epochs\d+_2conv_bi_\d+.ckpt')
p5 = re.compile(r'epochs\d+_2conv_bi_skip1_\d+.ckpt')
p6 = re.compile(r'epochs\d+_2conv_bi_skip2_\d+.ckpt')

p7 = re.compile(r'epochs\d+_3conv_uni_\d+.ckpt')
p8 = re.compile(r'epochs\d+_3conv_uni_skip1_\d+.ckpt')

p9 = re.compile(r'epochs\d+_3conv_bi_\d+.ckpt')
p10 = re.compile(r'epochs\d+_3conv_bi_skip1_\d+.ckpt')


data_frame_lists = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
pattern_lists = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
save_lists = ["2conv_uni_skip1", "2conv_uni_skip2", "2conv_uni", "2conv_bi_skip1", "2conv_bi_skip2", "2conv_bi",
              "3conv_uni_skip1", "3conv_uni", "3conv_bi_skip1", "3conv_bi"
              ]

for model_path in model_lists:
    for idx, pattern in enumerate(pattern_lists):
        if pattern.search(model_path):
            num_of_layers = 2 if idx < 6 else 3
            skip_steps = idx % 3
            cur_df = data_frame_lists[idx]
            res = get_adv_wer_adv_cer(model_path, test_path, cur_df)
            data_frame_lists[idx] = data_frame_lists[idx].append(res)


for idx, df in enumerate(data_frame_lists):
    filename = os.path.join('experiment_results2', save_lists[idx] + '_fgsm.csv')
    df.to_csv(filename)


