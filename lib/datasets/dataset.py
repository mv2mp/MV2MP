import os
import copy
import glob

import cv2
import hydra
import numpy as np
import torch
import pickle
from typing import Dict, List, Any, Literal

from reconstruction.vid2avatar.lib.utils import utils
from reconstruction.vid2avatar.lib.datasets import utils as dataset_utils


def get_cams(scene_path: str) -> Dict[Literal["K", "D", "R", "T"], List[np.ndarray]]:
    intri = cv2.FileStorage(f"{scene_path}/intri.yml", cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(f"{scene_path}/extri.yml", cv2.FILE_STORAGE_READ)
    cams = {"K": [], "D": [], "R": [], "T": []}
    for i in range(23):
        cams["K"].append(intri.getNode("K_Camera_B{}".format(i + 1)).mat())
        cams["D"].append(intri.getNode("dist_Camera_B{}".format(i + 1)).mat().T)
        cams["R"].append(extri.getNode("Rot_Camera_B{}".format(i + 1)).mat())
        cams["T"].append(extri.getNode("T_Camera_B{}".format(i + 1)).mat())
    return cams


def get_img_paths(scene_path: str) -> List[List[str]]:
    all_ims = []
    for i in range(23):
        i = i + 1
        data_root = f"{scene_path}/Camera_B{i}"
        ims = glob.glob(os.path.join(data_root, "*.jpg"))
        all_ims.append(sorted(ims))
    num_img = min([len(ims) for ims in all_ims])
    return [ims[:num_img] for ims in all_ims]


def get_mask_paths(scene_path: str) -> List[List[str]]:
    all_ims = []
    for i in range(23):
        i = i + 1
        data_root = f"{scene_path}/mask_cihp/Camera_B{i}"
        ims = glob.glob(os.path.join(data_root, "*.png"))
        all_ims.append(sorted(ims))
    num_img = min([len(ims) for ims in all_ims])
    return [ims[:num_img] for ims in all_ims]


class PickleDataset:
    def __init__(self, file_name, folder_path, scale=None):
        with open(os.path.join(folder_path, file_name), "rb") as file:
            self._cache = pickle.load(file)
        camera_set = set()
        frame_id_set = set()
        for key in self._cache.keys():
            camera_set.add(key[1])
            frame_id_set.add(key[0])

        self.camera_list = list(camera_set)
        self.camera_list.sort()
        self.frames_ids_list = list(frame_id_set)
        self.frames_ids_list.sort()
        self.scale = scale
        assert len(self.camera_list) * len(self.frames_ids_list) == len(self._cache), "Have Missing cameras"

    def __len__(self):
        return len(self.camera_list) * len(self.frames_ids_list)

    @staticmethod
    def _decompress_image_from_buffer(compressed_buffer: bytes) -> np.ndarray:
        compressed_data = np.frombuffer(compressed_buffer, np.uint8)
        decompressed_image = cv2.imdecode(compressed_data, cv2.IMREAD_UNCHANGED)
        return decompressed_image

    def map_id_to_camera_name(self, cam_id):
        return self.camera_list[cam_id]

    def map_frame_id_to_initial_frame_id(self, frame_id):
        return self.frames_ids_list[frame_id]

    def get_images(self, r):
        images = [r["image"][:, :, 0:3]]
        return images

    def get_masks(self, r):
        return [r["mask"]]

    def get_smpl_vertices(self, r):
        return r["smpl_vertices"]

    def get_world_to_smpl(self, r):
        return r["world_to_smpl"]

    def get_body_pose(self, r):
        return r["body_pose"]

    def get_betas(self, r):
        return r["betas"]

    def get_payload(self, frame_id, camera_id):
        dataset_frame_id = self.map_frame_id_to_initial_frame_id(frame_id)
        dataset_camera = self.map_id_to_camera_name(camera_id)
        cache_key = (dataset_frame_id, dataset_camera)
        cached = self._cache[cache_key]
        payload = copy.deepcopy(cached)
        return payload, dataset_frame_id, dataset_camera

    def get(self, cameras: List[int], frame_id: int) -> Dict[str, Any]:
        payload, dataset_frame_id, dataset_camera = self.get_payload(frame_id, cameras[0])

        payload["image"] = PickleDataset._decompress_image_from_buffer(payload["image"])
        payload["mask"] = PickleDataset._decompress_image_from_buffer(payload["mask"])

        image = self.get_images(payload)[0]
        mask = self.get_masks(payload)[0]
        intrinsics = payload["intrinsics"]
        if self.scale is not None:
            # with scale i should change masks, images, intrinsics.
            new_height = int(image.shape[0] * self.scale)
            new_width = int(image.shape[1] * self.scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            intrinsics[:2, :] *= self.scale

        return {
            "images": [image],
            "masks": [mask],
            "smpl_vertices": self.get_smpl_vertices(payload),
            "extrinsics": [payload["extrinsics"]],
            "intrinsics": [intrinsics],
            "world_to_smpl": self.get_world_to_smpl(payload),
            "smpl_tr": None,
            "smpl_rh": None,
            "poses": self.get_body_pose(payload),
            "shapes": self.get_betas(payload),
            "dataset_frame_id": dataset_frame_id,
            "dataset_camera": dataset_camera,
        }

    def read_corrected_smpl_poses(self):
        return self

    def __getitem__(self, key):
        """
        This is crutch that is used to obtain correct smpl params. Should be replaced after.
        """
        frame_id = key
        r, _, _ = self.get_payload(frame_id, 0)
        scale = np.array([1.0])
        world_to_smpls = self.get_world_to_smpl(r)
        smpl_poses = self.get_body_pose(r)
        smpl_shapes = self.get_betas(r)
        results = []

        for person_id, world_to_smpl in enumerate(world_to_smpls):
            smpl_trans, global_orient = dataset_utils.extract_pose_and_rodrigues(np.linalg.inv(world_to_smpl))
            smpl_pose = smpl_poses[person_id]
            smpl_shape = smpl_shapes[person_id]
            result = np.concatenate([scale, smpl_trans, global_orient, smpl_pose, smpl_shape], axis=0)
            results.append(result)
        return np.array(results)


class FrameRange:
    def __init__(self, from_inc, to_inc):
        self.from_inc = from_inc
        self.to_inc = to_inc

    def in_range(self, dataset_frame_id) -> bool:
        return (dataset_frame_id >= self.from_inc) and (dataset_frame_id <= self.to_inc)


class PickleV2aDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split, smpl_scale=1.0, img_scale=None):
        self.pickle_dataset = PickleDataset(metainfo.subject, metainfo.data_dir, img_scale)
        self.smpl_scale = smpl_scale
        self.frames_count = len(self.pickle_dataset.frames_ids_list)
        assert (
            metainfo.frames_count == self.frames_count
        ), "model reserved space for frames is not consistent with dataset"
        self.n_images = len(self.pickle_dataset)  # cam1, cam2, etc.
        # other properties
        self.num_sample = split.num_sample
        self.sampling_strategy = "weighted"
        self.img_size = self.pickle_dataset.get([0], 0)["images"][0].shape[:2]
        self.pose_corrected = self.pickle_dataset.read_corrected_smpl_poses()
        self.colors = np.random.rand(self.frames_count, 3)
        self.banned_cameras_dataset_cameras = set(metainfo.banned_cameras_dataset_cameras)
        self.frame_ranges = [
            FrameRange(from_inc=frame_range[0], to_inc=frame_range[1])
            for frame_range in metainfo.frames_ids_ranges_to_train
        ]

        zju_cache = {}
        if split.cache:
            for idx in range(self.n_images):
                zju_cache[idx] = self.get_initial_info(idx)
        self.zju_cache = zju_cache

    def get_initial_info(self, idx):
        # idx - это id кадра начиная с 0, не учитывая сдвиг, не учитывая того что камеры могут быть разные.

        camera_num = idx // self.frames_count
        frame_num = idx % self.frames_count
        zju_res = self.pickle_dataset.get([camera_num], frame_num)
        zju_res["pose_corrected"] = self.pose_corrected[(frame_num)]
        return zju_res

    def __len__(self):
        return self.n_images

    def should_skip(self, cache_item) -> bool:
        dataset_camera = cache_item["dataset_camera"]
        dataset_frame_id = cache_item["dataset_frame_id"]

        if dataset_camera in self.banned_cameras_dataset_cameras:
            return True

        in_train_range = False
        for frame_range in self.frame_ranges:
            if frame_range.in_range(dataset_frame_id):
                in_train_range = True
                break
        return not in_train_range

    def __getitem__(self, idx):
        frame_num = idx % self.frames_count
        if idx in self.zju_cache:
            zju_res = self.zju_cache[idx]
        else:
            zju_res = self.get_initial_info(idx)

        skip = self.should_skip(zju_res)

        img = zju_res["images"][0] / 255
        mask = zju_res["masks"][0]

        # img[mask==0] = self.colors[frame_num] # do it because we force

        extrinsics = zju_res["extrinsics"][0]
        extr_4x4 = np.eye(4)
        extr_4x4[0:3, :] = extrinsics
        pose_4x4 = np.linalg.inv(extr_4x4)
        cam_pose = pose_4x4[:3, :]
        intrinsics = zju_res["intrinsics"][0]
        world_to_smpl = zju_res["world_to_smpl"]
        # no_rays_mask = zju_res['no_rays_mask'][0]

        #### начинаем готовить датасет ####
        img_size = img.shape[:2]
        self.img_size = img_size
        uv = np.mgrid[: img_size[0], : img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        smpl_params = torch.from_numpy(zju_res["pose_corrected"]).float()

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "foreground_mask": mask,
            }

            samples, index_outside, sample_fg_mask, _ = utils.weighted_sampling(
                data, img_size, self.num_sample, bbox_ratio=1
            )
            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "pose": cam_pose.astype(np.float32),
                "smpl_params": smpl_params,
                "index_outside": index_outside.astype(np.int32),
                "idx": idx,
                "frame_num": frame_num,
                "camera_name": "__sentinel__",
                "world_to_smpl": world_to_smpl,
                "skip": skip,
            }
            images = {"rgb": samples["rgb"].astype(np.float32), "sample_fg_mask": sample_fg_mask}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "pose": cam_pose.astype(np.float32),
                "smpl_params": smpl_params,
                "idx": idx,
                "frame_num": frame_num,
                "camera_name": "__sentinel__",
                "world_to_smpl": world_to_smpl,
                "skip": skip,
            }
            images = {"rgb": img.reshape(-1, 3).astype(np.float32), "img_size": self.img_size}
            return inputs, images


class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        root = os.path.join("../data", metainfo.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.start_frame = metainfo.start_frame
        self.end_frame = metainfo.end_frame
        self.skip_step = 1
        self.images, self.img_sizes = [], []
        self.training_indices = list(range(metainfo.start_frame, metainfo.end_frame, self.skip_step))

        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        # only store the image paths to avoid OOM
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.n_images = len(self.img_paths)

        # coarse projected SMPL masks, only for sampling
        mask_dir = os.path.join(root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))  # - маски нужны для тренировки чтоы лучше семплировать
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))  # betas of smpl
        self.poses = np.load(os.path.join(root, "poses.npy"))[self.training_indices]  # it is tetas = 23 * 3 + 3 = 72
        self.trans = np.load(os.path.join(root, "normalize_trans.npy"))[
            self.training_indices
        ]  # это похоже смещение относительно мира
        # cameras
        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [
            camera_dict["scale_mat_%d" % idx].astype(np.float32) for idx in self.training_indices
        ]  # это в итоге просто интринзики так задаются.
        world_mats = [
            camera_dict["world_mat_%d" % idx].astype(np.float32) for idx in self.training_indices
        ]  # тут вроде понятно, это экстринзики - world to camera в общем

        self.scale = 1 / scale_mats[0][0, 0]  # а так же это параметр

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all)

        # other properties
        self.num_sample = split.num_sample
        self.sampling_strategy = "weighted"

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # normalize RGB
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255

        mask = cv2.imread(self.mask_paths[idx])
        # preprocess: BGR -> Gray -> Mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0

        img_size = self.img_size

        uv = np.mgrid[: img_size[0], : img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "foreground_mask": mask,
            }

            samples, index_outside, _, _ = utils.weighted_sampling(data, img_size, self.num_sample)
            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "index_outside": index_outside,
                "idx": idx,
                "frame_num": idx,
                "camera_name": "__sentinel__",
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "idx": idx,
                "frame_num": idx,
                "camera_name": "__sentinel__",
            }
            images = {"rgb": img.reshape(-1, 3).astype(np.float32), "img_size": self.img_size}
            return inputs, images


class PickleValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = PickleV2aDataset(metainfo, split, img_scale=0.5)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        while True:
            image_id = int(np.random.choice(len(self.dataset), 1))
            self.data = self.dataset[image_id]
            inputs, images = self.data
            if inputs["skip"] == False:
                break

        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "smpl_params": inputs["smpl_params"],
            "image_id": image_id,
            "idx": inputs["idx"],
            "frame_num": inputs["frame_num"],
            "camera_name": "__sentinel__",
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            "pixel_per_batch": self.pixel_per_batch,
            "total_pixels": self.total_pixels,
        }
        return inputs, images


class PickleTestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = PickleV2aDataset(metainfo, split)

        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return self.dataset.frames_count

    def __getitem__(self, idx):
        data = self.dataset[idx]

        inputs, images = data
        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "smpl_params": inputs["smpl_params"],
            "idx": inputs["idx"],
            "frame_num": inputs["frame_num"],
            "camera_name": "__sentinel__",
        }
        images = {"rgb": images["rgb"], "img_size": images["img_size"]}
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "smpl_params": inputs["smpl_params"],
            "image_id": image_id,
            "idx": inputs["idx"],
            "camera_name": "__sentinel__",
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            "pixel_per_batch": self.pixel_per_batch,
            "total_pixels": self.total_pixels,
        }
        return inputs, images


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)

        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        inputs, images = data
        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs["intrinsics"],
            "pose": inputs["pose"],
            "smpl_params": inputs["smpl_params"],
            "idx": inputs["idx"],
            "camera_name": "__sentinel__",
        }
        images = {"rgb": images["rgb"], "img_size": images["img_size"]}
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx
