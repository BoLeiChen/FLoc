import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import wandb
import matplotlib
matplotlib.use('Agg')
def visualize_floorplan_rays(
    model,
    obs_image,
    floorplan_image,
    pose,
    gt_ray,
    wh_tensor,
    rmd_matrix,
    config,
    device,
    epoch,
    sample_idx,
):
    try:
        # Data preparation
        gt_ray_k = gt_ray
        obs_image_k = obs_image.unsqueeze(0).to(device)
        rmd_matrix_k = rmd_matrix.unsqueeze(0).to(device)
        
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        obs_image_k = transform(obs_image_k)
        
        floorplan_np = floorplan_image.cpu().numpy().transpose(1, 2, 0)
        if floorplan_np.shape[2] == 1:
            floorplan_np = floorplan_np.squeeze(2)
        floorplan_np = np.ascontiguousarray(floorplan_np)

        pose_k = pose.cpu().numpy()
        
        with torch.no_grad():
            features = model("encode", obs_img=obs_image_k, rmd_matrix=rmd_matrix_k)
            pred_d = model("decoder_inference", depth_cond=features, num_samples=1)
           

        gt_ray_np = gt_ray_k.squeeze().cpu().numpy()
        pred_ray_np = pred_d.squeeze().cpu().numpy()

        # Coordinates calculation
        pixels_per_meter = 1 / 0.01
        w0, h0 = wh_tensor[0].item(), wh_tensor[1].item()
        h_resized, w_resized = floorplan_np.shape[:2]
        
        agent_x_metric, agent_y_metric, yaw = pose_k[0], pose_k[1], pose_k[2]
        
        # Map metric position to resized image pixel
        # (Assuming GT pose is relative to center in original resolution)
        scale_x = w_resized / w0
        scale_y = h_resized / h0
        
        # If pose is top-left based (training code logic):
        # agent_pos_orig_pix = np.array([agent_x_metric, agent_y_metric]) * pixels_per_meter
        
        # If pose is center based (your previous viz logic):
        agent_pos_orig_pix = np.array([agent_x_metric, agent_y_metric]) * pixels_per_meter + np.array([w0 / 2, h0 / 2])
        
        agent_x_pix = agent_pos_orig_pix[0] * scale_x
        agent_y_pix = agent_pos_orig_pix[1] * scale_y

        ray_n = 40
        fov_rad = np.deg2rad(108)
        focal_length_pix = (ray_n / 2) / np.tan(fov_rad / 2)
        pixel_coords_y = np.arange(ray_n) - (ray_n - 1) / 2
        center_angs = np.flip(np.arctan2(pixel_coords_y, focal_length_pix))
        ray_angles = yaw + center_angs

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(floorplan_np, cmap='gray')

        for j in range(40):
            # GT Ray (Green)
            gt_dist = gt_ray_np[j]
            gt_end_x_m = agent_x_metric + gt_dist * np.cos(ray_angles[j])
            gt_end_y_m = agent_y_metric + gt_dist * np.sin(ray_angles[j])
            
            gt_end_pix_orig = np.array([gt_end_x_m, gt_end_y_m]) * pixels_per_meter + np.array([w0 / 2, h0 / 2])
            gt_end_pix_x = gt_end_pix_orig[0] * scale_x
            gt_end_pix_y = gt_end_pix_orig[1] * scale_y
            
            ax.plot([agent_x_pix, gt_end_pix_x], [agent_y_pix, gt_end_pix_y], 'g-', linewidth=0.5, alpha=0.6)

            # Pred Ray (Red)
            pred_dist = pred_ray_np[j]
            pred_end_x_m = agent_x_metric + pred_dist * np.cos(ray_angles[j])
            pred_end_y_m = agent_y_metric + pred_dist * np.sin(ray_angles[j])
            
            pred_end_pix_orig = np.array([pred_end_x_m, pred_end_y_m]) * pixels_per_meter + np.array([w0 / 2, h0 / 2])
            pred_end_pix_x = pred_end_pix_orig[0] * scale_x
            pred_end_pix_y = pred_end_pix_orig[1] * scale_y
            
            ax.plot([agent_x_pix, pred_end_pix_x], [agent_y_pix, pred_end_pix_y], 'r-', linewidth=0.5, alpha=0.8)

        ax.add_artist(plt.Circle((agent_x_pix, agent_y_pix), radius=3, color='blue'))
        ax.set_title(f'Sample {sample_idx} - Ray Viz (Epoch {epoch})')
        ax.axis('off')
        
        wandb.log({f"Validation/Floorplan_Ray_Viz_{sample_idx}": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate ray visualization for sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()


class ImageLoggerCallback(pl.Callback):
    def __init__(self, num_images_log=8, image_log_freq=1000):
        super().__init__()
        self.num_images_log = num_images_log
        self.image_log_freq = image_log_freq

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        
        if (trainer.global_step + 1) % self.image_log_freq != 0:
            return

        if not pl_module.config.get("use_wandb", False) or not trainer.is_global_zero:
            return

        (
            obs_image,
            pose,
            ray,
            floorplan_image,
            wh_tensor,
            rmd_matrix
        ) = batch

        device = pl_module.device
        num_to_log = min(self.num_images_log, obs_image.shape[0])
        
        pl_module.eval()
        
        with torch.no_grad():
            for k in range(num_to_log):
                visualize_floorplan_rays(
                    model=pl_module.model,
                    obs_image=obs_image[k],
                    floorplan_image=floorplan_image[k],
                    pose=pose[k],
                    gt_ray=ray[k],
                    wh_tensor=wh_tensor[k],
                    rmd_matrix=rmd_matrix[k],
                    config=pl_module.config,
                    device=device,
                    epoch=trainer.current_epoch,
                    sample_idx=k,
                )
        
        pl_module.train()