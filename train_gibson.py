import argparse
import yaml
from attrdict import AttrDict

from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *

from torch.utils.data import DataLoader
import lightning as Li
from lightning.pytorch.callbacks import ModelCheckpoint


def train_observation():
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
        choices=["gibson_f", "gibson_g"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/Gibson Floorplan Localization Dataset",
        help="path of the dataset",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./logs", help="path of the checkpoints"
    )
    args = parser.parse_args()

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # network to evaluate
    net_type = args.net_type

    # parameters
    L = 3  # number of the source frames
    D = 128  # number of depth planes
    d_min = 0.1  # minimum depth
    d_max = 15.0  # maximum depth
    d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # camera intrinsic, focal length / image width
    trans_thresh = 0.005  # translation threshold (variance) if using comp_s

    add_rp = (
        True  # whether use roll and pitch angle augmentation, only used in training
    )
    roll = 0  # maximum roll augmentation in randian
    pitch = 0  # maximum pitch augmentation in randian

    # paths
    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    depth_dir = dataset_dir
    log_dir = args.ckpt_path
    desdf_path = os.path.join(args.dataset_path, "desdf")

    if net_type == "d":
        depth_suffix = "depth40"
    else:
        depth_suffix = "depth160"
    #
    # depth_suffix = "depth40"

    # instanciate dataset
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    train_set = GridSeqDataset(
        dataset_dir,
        split.train,
        L=L,
        depth_dir=depth_dir,
        depth_suffix=depth_suffix,
        add_rp=add_rp,
        roll=roll,
        pitch=pitch,
    )

    val_set = GridSeqDataset(
        dataset_dir,
        split.val,
        L=L,
        depth_dir=depth_dir,
        depth_suffix=depth_suffix,
        add_rp=add_rp,
        roll=roll,
        pitch=pitch,
    )

    # create model
    if net_type == "mvd":
        net = mv_depth_net_pl(
            D=D,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            shape_loss_weight = True,
        ).to(device)
    if net_type == "d":
        net = depth_net_pl(
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
            shape_loss_weight=True,
        ).to(device)
    if net_type == "comp":
        net = comp_d_net_pl(
            mv_net=mv_depth_net_pl(D=D, d_hyp=d_hyp).net,
            mono_net=depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D).encoder,
            L=L,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
            F_W=F_W,
            use_pred=True,
            shape_loss_weight=True,
        ).to(device)

    # =====================
    # ==== TRAINING =====
    # =====================
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=6)
    # val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)
    checkpoint_callback = ModelCheckpoint(
        #monitor='loss-train',
        dirpath="comp_f",
        #filename='sample-{epoch:02d}-{loss-train:.2f}',
        every_n_epochs=1,
        save_top_k=-1
    )
    trainer = Li.Trainer(max_epochs=100, gpus=1, callbacks=[checkpoint_callback])
    # trainer.fit(net, train_loader, val_loader)
    trainer.fit(net, train_loader)

if __name__ == "__main__":
    train_observation()
