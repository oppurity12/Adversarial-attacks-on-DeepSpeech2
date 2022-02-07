import hydra
import torch
from typing import Union

from deepspeech_attack.configs.attack_config import AttackConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder


from deepspeech_attack.attacks.fgsm import fg_main
from deepspeech_attack.attacks.pgd import pgd_main


def run(cfg: AttackConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")
    attack_dir = {"fgsm": fg_main, 'pgd': pgd_main}

    model = load_model(
        device=device,
        model_path=cfg.model.model_path,
    )

    test_model = load_model(
        device=device,
        model_path=cfg.model.model_path,
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    test_dataset = SpectrogramDataset(
        audio_conf=model.spect_cfg,
        input_path=hydra.utils.to_absolute_path(cfg.test_path),
        labels=model.labels,
        normalize=True
    )
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    attack = attack_dir[cfg.attack_type]
    wer, cer = attack(
        test_loader=test_loader,
        device=device,
        model=model,
        test_model=test_model,
        decoder=decoder,
        target_decoder=target_decoder,
        precision=cfg.model.precision,
        cfg=cfg
    )

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))

    return wer, cer
