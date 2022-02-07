import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.save_config import SaveConfig
from save_results import save_result

cs = ConfigStore.instance()
cs.store(name="config", node=SaveConfig)


@hydra.main(config_path=None, config_name="config")
def hydra_main(cfg: SaveConfig):
    save_result(cfg=cfg, adv_save=False)
    if cfg.adv_save:
        save_result(cfg=cfg, adv_save=True)  # adv_results도 저장


if __name__ == '__main__':
    hydra_main()

