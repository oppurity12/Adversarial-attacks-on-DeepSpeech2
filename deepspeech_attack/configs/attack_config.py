from dataclasses import dataclass
import sys
from types import FunctionType

sys.path.append("/home/skku/ML/deepspeech.pytorch/")

from deepspeech_pytorch.configs.inference_config import EvalConfig


@dataclass
class AttackConfig(EvalConfig):
    """
    attack_type: FGSM, PGD, CW

    FGSM일때 epsilon inputs
    """

    batch_size: int = 20
    attack_type: str = ''
    shortcut_cfg: bool = False
    attack_steps: int = 5
    epsilon: float = 0

