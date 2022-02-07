import os.path
from glob import glob
import pandas as pd
import re

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.testing import evaluate
from deepspeech_attack.configs.attack_config import AttackConfig
from deepspeech_attack.attacks.attack import run
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.save_config import SaveConfig


def get_wer_cer(model_path, test_path, df, epsilon=0.1, attack_type=None):
    cfg = EvalConfig()
    cfg.model.model_path = model_path
    model_path = os.path.basename(model_path)
    cfg.test_path = test_path
    wer, cer = evaluate(cfg)
    data = [[model_path, wer, cer]]
    result = pd.DataFrame(data, columns=['model_path', 'wer', 'cer'])
    df = df.append(result)
    return df


def get_adv_wer_adv_cer(model_path, test_path, df, epsilon=0.1, attack_type='fgsm'):
    cfg = AttackConfig()
    cfg.model.model_path = model_path
    cfg.test_path = test_path
    cfg.epsilon = epsilon
    cfg.attack_type = attack_type
    wer, cer = run(cfg)
    model_path = os.path.basename(model_path)
    data = [[model_path, wer, cer, epsilon]]
    result = pd.DataFrame(data, columns=['model_path', 'adv_wer', 'adv_cer', 'epsilon'])
    df = df.append(result)
    return df


def save_result(cfg: SaveConfig, adv_save=False):
    checkpoint_dir = to_absolute_path(cfg.checkpoint_dir)
    save_dir = to_absolute_path(cfg.save_dir)
    test_path = to_absolute_path(cfg.test_path)

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError("checkpoint_dir doesn't exists")
    model_lists = list(glob(os.path.join(checkpoint_dir, '*.ckpt')))

    get_func = get_adv_wer_adv_cer if adv_save else get_wer_cer
    suffix = "_fgsm.csv" if adv_save else '.csv'

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

    df11 = pd.DataFrame()  # 4conv, bi skip1
    df12 = pd.DataFrame()  # 4conv, bi skip2
    df13 = pd.DataFrame()  # 4conv, bi

    df14 = pd.DataFrame()  # 2conv, bi skip1 runge

    p1 = re.compile(r'epochs\d+_2conv_uni_skip1_\d+.ckpt')
    p2 = re.compile(r'epochs\d+_2conv_uni_skip2_\d+.ckpt')
    p3 = re.compile(r'epochs\d+_2conv_uni_\d+.ckpt')

    p4 = re.compile(r'epochs\d+_2conv_bi_skip1_\d+.ckpt')
    p5 = re.compile(r'epochs\d+_2conv_bi_skip2_\d+.ckpt')
    p6 = re.compile(r'epochs\d+_2conv_bi_\d+.ckpt')

    p7 = re.compile(r'epochs\d+_3conv_uni_skip1_\d+.ckpt')
    p8 = re.compile(r'epochs\d+_3conv_uni_\d+.ckpt')

    p9 = re.compile(r'epochs\d+_3conv_bi_skip1_\d+.ckpt')
    p10 = re.compile(r'epochs\d+_3conv_bi_\d+.ckpt')

    p11 = re.compile(r'epochs\d+_4conv_bi_skip1_\d+.ckpt')
    p12 = re.compile(r'epochs\d+_4conv_bi_skip2_\d+.ckpt')
    p13 = re.compile(r'epochs\d+_4conv_bi_\d+.ckpt')

    p14 = re.compile(r'epochs\d+_2conv_bi_skip1_runge_\d+.ckpt')

    data_frame_lists = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]
    pattern_lists = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]
    save_lists = ["2conv_uni_skip1", "2conv_uni_skip2", "2conv_uni", "2conv_bi_skip1", "2conv_bi_skip2", "2conv_bi",
                  "3conv_uni_skip1", "3conv_uni", "3conv_bi_skip1", "3conv_bi", "4conv_bi_skip1", "4conv_bi_skip2",
                  "4conv_bi", "2conv_bi_skip1_runge"
                  ]

    for model_path in model_lists:
        for idx, pattern in enumerate(pattern_lists):
            if pattern.search(model_path):
                try:
                    cur_df = data_frame_lists[idx]
                    res = get_func(model_path, test_path, cur_df, epsilon=cfg.epsilon, attack_type=cfg.attack_type)
                    data_frame_lists[idx] = res
                except FileNotFoundError:
                    raise FileNotFoundError(f"something wrong with {model_path}")

    for idx, df in enumerate(data_frame_lists):
        filename = os.path.join(save_dir, save_lists[idx] + suffix)
        df.to_csv(filename)
