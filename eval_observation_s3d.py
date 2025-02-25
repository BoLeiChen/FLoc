import argparse
import os

import torch
import tqdm
import yaml
from attrdict import AttrDict

from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils_s3d import *
from utils.localization_utils import *


def evaluate_observation():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--net_type",
        type=str,
        default="comp",
        choices=[
            "d",
            "mvd",
            "comp",
            "comp_s",
        ],  # d: monocualr, mvd: multi-view, comp: learned complementary, comp_s: hard threshold complementary
        help="type of the network to evaluate. d: monocualr, mvd: multi-view, comp: learned complementary, comp_s: hard threshold complementary",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gibson_g",
        choices=["gibson_f", "gibson_g", "Structured3D"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/s3d",
        help="path of the dataset",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./d_s3d", help="path of the checkpoints"
    )
    args = parser.parse_args()

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # parameters
    L = 3  # number of the source frames
    D = 128  # number of depth planes
    d_min = 0.1  # minimum depth
    d_max = 15.0  # maximum depth
    d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # camera intrinsic, focal length / image width
    trans_thresh = 0.005  # translation threshold (variance) if using comp_s

    # paths
    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    depth_dir = dataset_dir
    log_dir = args.ckpt_path
    desdf_path = os.path.join(args.dataset_path, "desdf")

    test_set = S3DDataset(
        dataset_dir,
        [3249, 3500],
    )

    d_net = depth_net_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(log_dir, "mono.ckpt"),
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
        ).to(device)

    # =====================
    # ==== EVALUATION =====
    # =====================

    # get desdf for the scene
    print("load desdf ...")
    desdfs = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 20] = 20  # truncate

    # get the ground truth poses
    print("load poses and maps ...")
    maps = {}
    gt_poses = {}

    for scene in tqdm.tqdm(test_set.scene_names):
        # load map
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h = occ.shape[0]
        w = occ.shape[1]

        # get poses
        with open(os.path.join(dataset_dir, scene, "poses_map.txt"), "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]
            traj_len = len(poses_txt)
            poses = np.zeros([traj_len, 3], dtype=np.float32)
            for state_id in range(traj_len):
                pose = poses_txt[state_id].split(" ")
                # from world coordinate to map coordinate
                # x = float(pose[0]) / 1000 / 0.02 + w / 2
                # y = float(pose[1]) / 1000 / 0.02 + h / 2
                x = pose[0]
                y = pose[1]
                th = float(pose[2])
                poses[state_id, :] = np.array((x, y, th), dtype=np.float32)

            gt_poses[scene] = poses

    # record the accuracy
    acc_record = []
    acc_orn_record = []
    for data_idx in tqdm.tqdm(range(len(test_set))):
        data = test_set[data_idx]
        # get the scene name according to the data_idx
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        # get desdf
        desdf = desdfs[scene]

        # get reference pose in map coordinate and in scene coordinate
        ref_pose_map = gt_poses[scene][idx_within_scene, :]

        # transform to desdf frame
        gt_pose_desdf = ref_pose_map.copy()
        gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf["l"]) / 5
        gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf["t"]) / 5

        ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
        ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)

        pred_depths, attn_2d, prob = d_net.encoder(ref_img_torch, ref_mask_torch)
        pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
        pred_rays = get_ray_from_depth(pred_depths)
        pred_rays = torch.tensor(pred_rays, device="cpu")

        # localize with the desdf using the prediction
        prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize(
            torch.tensor(desdf["desdf"]), pred_rays
        )

        # calculate accuracy
        acc = np.linalg.norm(pose_pred[:2] - gt_pose_desdf[:2], 2.0) * 0.02
        acc_record.append(acc)
        acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
        acc_orn_record.append(acc_orn)

    acc_record = np.array(acc_record)
    acc_orn_record = np.array(acc_orn_record)
    print("1m recall = ", np.sum(acc_record < 1) / acc_record.shape[0])
    print("0.5m recall = ", np.sum(acc_record < 0.5) / acc_record.shape[0])
    print("0.1m recall = ", np.sum(acc_record < 0.1) / acc_record.shape[0])
    print(
        "1m 30 deg recall = ",
        np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30))
        / acc_record.shape[0],
    )


if __name__ == "__main__":
    evaluate_observation()

