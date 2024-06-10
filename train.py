import os
import glob
import logging

import yaml
import hydra
import hashlib
import omegaconf as oc
import pytorch_lightning as pl

from reconstruction.helpers import training_context as context

from reconstruction.vid2avatar.v2a_model import V2AModel
from reconstruction.vid2avatar.lib.datasets import create_dataset, create_dataloader
from reconstruction.vid2avatar.lib.utils import log, callbacks


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="base_hi4d_test_wo_merge")
def main(opt):
    pl.seed_everything(42)

    ctx = context.create_context(
        model_name=f"v2a",
        data_revision="v2a-data",
        comment=f"exp={opt.exp}",
        launch_file=__file__,
    )

    checkpoint_dir = os.path.join(ctx.state_dir, "checkpoints")

    config_dict = oc.omegaconf.OmegaConf.to_container(opt)

    config_str = yaml.dump(config_dict)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    with open(f"{ctx.state_dir}/config.yaml", "w") as f:
        f.write(config_str)

    callback_list = [
        callbacks.NirvanaCheckpointCallback(
            dirpath=checkpoint_dir,
            filename="{epoch:04d}-{loss}",
            save_on_train_epoch_end=True,
            save_top_k=1,
            save_last=True,
        )
    ]

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=callback_list,
        max_epochs=opt.trainer.max_epochs,
        check_val_every_n_epoch=opt.trainer.check_val_every_n_epoch,
        logger=log.PulsarTensorboardLogger(),
        log_every_n_steps=opt.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
    )

    train_loader = create_dataloader(create_dataset(opt.dataset.metainfo, opt.dataset.train), opt.dataset.train)
    valid_loaders = [
        create_dataloader(create_dataset(opt.dataset.metainfo, config), config) for config in opt.dataset.valid
    ]

    model = V2AModel(opt)

    checkpoint = None
    if opt.model.is_continue:
        if os.path.exists(checkpoint_dir) and len(glob.glob(f"{checkpoint_dir}/*.ckpt")):
            checkpoint = sorted(glob.glob(f"{checkpoint_dir}/*.ckpt"))[-1]
        else:
            ctx.logger.info("opt.model.is_continue = True, but no checkpoints found. Starting from scratch.")

    trainer.fit(model, train_loader, valid_loaders, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
