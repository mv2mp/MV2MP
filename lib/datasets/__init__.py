from torch.utils.data import DataLoader, Dataset

from reconstruction.vid2avatar.lib.datasets.ffcache import FFCacheWrapper
from reconstruction.vid2avatar.lib.datasets.dataset import (
    Dataset,
    ValDataset,
    TestDataset,
    PickleV2aDataset,
    PickleValDataset,
    PickleTestDataset,
)


def find_dataset_using_name(name):
    mapping = {
        "Video": Dataset,
        "VideoVal": ValDataset,
        "VideoTest": TestDataset,
        "VideoZju": PickleV2aDataset,
        "VideoZjuVal": PickleValDataset,
        "VideoZjuTest": PickleTestDataset,
        "FFCache": FFCacheWrapper,
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}")
    return cls


def create_dataset(metainfo, split):
    dataset_cls = find_dataset_using_name(split.type)
    return dataset_cls(metainfo, split)


def create_dataloader(
    dataset: Dataset,
    split,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=split.batch_size,
        drop_last=split.drop_last,
        shuffle=split.shuffle,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
