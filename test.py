import os
import glob

import yaml
import hydra
import omegaconf as oc
import pytorch_lightning as pl

import training_context as context

from v2a_model import V2AModel
from lib.datasets import create_dataset, create_dataloader
from lib.utils import log


@hydra.main(config_path="configs", config_name="base_hi4d_ff_test")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    ctx = context.create_context(
        model_name=f"v2a",
        data_revision="v2a-data",
        comment=f"exp={opt.exp}",
        launch_file=__file__,
    )

    model = V2AModel(opt)

    with open(f"{ctx.state_dir}/config.yaml", "w") as f:
        f.write(yaml.dump(oc.omegaconf.OmegaConf.to_container(opt)))

    logger = log.PulsarTensorboardLogger()

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        max_epochs=8000,
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    checkpoint_dir = os.path.join(ctx.state_dir, "checkpoints")
    assert os.path.exists(checkpoint_dir)

    checkpoint = sorted(glob.glob(f"{checkpoint_dir}/*.ckpt"))[-1]
    assert os.path.exists(checkpoint)

    test_loaders = [
        create_dataloader(create_dataset(opt.dataset.metainfo, config), config) for config in opt.dataset.valid
    ]
    trainer.test(model, test_loaders, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
