import hydra
from hydra.core.config_store import ConfigStore

from configs.attack_config import AttackConfig
from attacks.attack import run

cs = ConfigStore.instance()
cs.store(name="config", node=AttackConfig)


@hydra.main(config_path=None, config_name="config")
def hydra_main(cfg: AttackConfig):
    run(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
