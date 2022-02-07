from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class SaveConfig:
    checkpoint_dir: str = ""
    save_dir: str = ""
    test_path: str = ""
    adv_save: bool = True
    epsilon: float = 0.1
    attack_type: str = 'fgsm'

    # checkpoint_dir = 'checkpoints2'
    # save_dir = 'experiment_results4'
    # test_path = '/home/skku/ML/deepspeech.pytorch/data/an4_test_manifest.json'

