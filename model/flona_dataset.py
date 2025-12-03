import os
import sys
from typing import List, Tuple
import yaml
from pathlib import Path
import numpy as np
import tqdm
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

from .data_utils import (
    img_path_to_data,
    img_path_to_data_and_point_transfer,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class flona_Dataset(Dataset):
    def __init__(self, data_folder: str, data_splits_path: str, split: str, rgb_image_size: Tuple[int, int], floorplan_img_size: Tuple[int, int],
                 rmd_ray_num: int, rmd_m1: int, rmd_m2: int):
        self.data_folder = data_folder
        self.data_splits_path = data_splits_path
        self.split = split
        self.rgb_image_size = rgb_image_size
        self.floorplan_img_size = floorplan_img_size
        with open(self.data_splits_path, 'r', encoding='utf-8') as f:
            data_splits = yaml.safe_load(f)
        self.data_split = data_splits[self.split]
        self.data = self._load_data(self.data_folder, self.data_split)
        
        self.rmd_ray_num = rmd_ray_num
        self.rmd_m1 = rmd_m1
        self.rmd_m2 = rmd_m2
        
    def _load_data(self, data_folder, data_split):
        data = []
        for i in range(len(data_split)):
            cur_dir = os.path.join(data_folder, data_split[i])
            # 一行一个列表，元素为 float
            pose_data = [
                list(map(float, line.split()))  # 默认空格/Tab 分隔
                for line in Path(cur_dir + "/poses.txt").read_text(encoding='utf-8').splitlines()
                if line.strip()  # 跳过空行
            ]
            ray_data = [
                list(map(float, line.split()))  # 默认空格/Tab 分隔
                for line in Path(cur_dir + "/depth40.txt").read_text(encoding='utf-8').splitlines()
                if line.strip()  # 跳过空行
            ]
            rgb_dir = os.path.join(cur_dir, "rgb")

            def sort_key(p: Path):
                """返回 (前缀数字, 后缀数字) 二元组"""
                stem = p.stem  # '0000-0'
                a, b = map(int, stem.split('-'))
                return a, b

            files = sorted((p for p in Path(rgb_dir).iterdir() if p.is_file() and (p.suffix == '.png' or p.suffix == '.jpg')),
                           key=sort_key)
            rgb_names = [f.name for f in files]
            rmd_path = os.path.join(cur_dir, "map_rmd_140.npy")

            for n in range(len(rgb_names)):
                metadata = {}
                metadata["rgb_image"] = rgb_dir + "/" + rgb_names[n]
                metadata["floorplan_image"] = cur_dir + "/map.png"
                metadata["pose"] = pose_data[n]
                metadata["ray"] = ray_data[n]
                metadata["rmd_path"] = rmd_path
                data.append(metadata)
        return data

    def _load_image(self, image_path, target_size):
        try:  # directedly load from disk
            with open(image_path, "rb") as f:
                result = img_path_to_data(f, target_size)
            return result
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _load_floorplan(self, floorplan_path):
        floorplan_img = cv2.imread(floorplan_path, 0)
        return floorplan_img

        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        data = self.data[i]
        # rgb_image = self._load_image(data["rgb_image"], self.rgb_image_size)
        rgb_image = Image.open(data["rgb_image"])
        transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])  
        rgb_image = transfrom(rgb_image)
        # rgb_tensor = self.transform(rgb_image)
        
        pose = torch.tensor(data["pose"])
        ray = torch.tensor(data["ray"])

        w, h = Image.open(data["floorplan_image"]).size
        wh_tensor = torch.tensor([w, h], dtype=torch.float32)
        # floorplan_img = self._load_floorplan(data["floorplan_image"])
        floorplan_img = self._load_image(data["floorplan_image"], self.floorplan_img_size)
        rmd_file_content = np.load(data["rmd_path"], allow_pickle=True).item()
        rmd_matrix = rmd_file_content["rmd_matrix"]
        crop_offset = rmd_file_content["crop_origin"]
        
        return (
            torch.as_tensor(rgb_image, dtype=torch.float32),
            torch.as_tensor(pose, dtype=torch.float32),
            torch.as_tensor(ray, dtype=torch.float32),
            torch.as_tensor(floorplan_img, dtype=torch.float32),
            torch.as_tensor(wh_tensor, dtype=torch.float32),
            torch.as_tensor(rmd_matrix, dtype=torch.float32),
        )
        
if __name__ == "__main__":
    DATA_FOLDER = 'datasets/gibson_f'
    DATA_SPLITS_PATH = 'datasets/gibson_f/data_splits.yaml'
    SPLIT = 'train'
    RGB_SIZE = (256, 256)
    
    dataset = flona_Dataset(
            data_folder=DATA_FOLDER,
            data_splits_path=DATA_SPLITS_PATH,
            split=SPLIT,
            rgb_image_size=RGB_SIZE,
            rmd_ray_num=8,
            rmd_m1=9,
            rmd_m2=9
        )