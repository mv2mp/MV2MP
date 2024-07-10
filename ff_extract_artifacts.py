import os
import glob
import logging

import yaml
import hydra
import omegaconf as oc
import pytorch_lightning as pl

from reconstruction.helpers import training_context as context

from reconstruction.vid2avatar.v2a_model import V2AModel
from reconstruction.vid2avatar.lib.datasets import create_dataset, create_dataloader
from reconstruction.vid2avatar.lib.utils import callbacks


logger = logging.getLogger(__name__)


@hydra.main()
def main(opt):
    device = "cuda"

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
    with open(f"{ctx.state_dir}/config.yaml", "w") as f:
        f.write(config_str)

    assert "artifact_extraction" in opt.dataset, "artifact_extraction section is required for mesh extraction"

    artifact_extraction_loader_config = opt.dataset.artifact_extraction
    assert "ff_config" in artifact_extraction_loader_config, "ff_config is required for artifact_extraction"
    assert not artifact_extraction_loader_config.shuffle, "shuffled dataloader is not supported"
    ff_config = artifact_extraction_loader_config.ff_config
    ff_path = ff_config.ff_path

    mesh_extraction_dataloader = create_dataloader(
        create_dataset(opt.dataset.metainfo, artifact_extraction_loader_config), artifact_extraction_loader_config
    )

    logger.info(f"Artifacts will be stored in: {ff_path}")

    extractor = callbacks.ArtifactsExportToFFCallback(
        opt.extractor, ff_path=ff_path, dataloader=mesh_extraction_dataloader
    )

    checkpoint = sorted(glob.glob(f"{checkpoint_dir}/*.ckpt"))[-1]
    model = V2AModel.load_from_checkpoint(checkpoint_path=checkpoint, opt=opt).eval()
    model = model.to(device)
    extractor.call(model)


if __name__ == "__main__":
    main()
