import random
import time
import os
import sys

import PIL.Image as Image
import igibson
import cv2
import networkx as nx  
import numpy as np
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

sys.path.append('/path/to/flodiff')
from igibson.simulator import Simulator
from igibson.render.profiler import Profiler
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from model.data_utils import img_path_to_data, resize_and_aspect_crop
from model.flona import flona, DenseNetwork
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from training.train_eval_loop import load_model
from training.train_utils import execute_model

IMG_RESIZE_SIZE = (96, 96)
IMAGE_ASPECT_RATIO = 1 / 1  # 4 / 3


def check_collision(pos, travers_map):
    """
    check if the position is in collision
    """
    # print(pos)
    pos_in_map = (pos * 100 + np.array([travers_map.shape[0], travers_map.shape[1]]) // 2).astype(np.int16)
    return travers_map[pos_in_map[1], pos_in_map[0]] == 0

def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def camera_set_and_record(env, current_position, current_heading):
    env.viewer.initial_pos = current_position
    env.viewer.initial_view_direction = current_heading - current_position # [0,0,0] will make the image black
    env.viewer.reset_viewer()
    env.renderer.set_camera(current_position, current_heading, [0, 0, 1])
    with Profiler("Render"):
        frame = env.renderer.render(modes=("rgb"))
    img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
    # plt.imshow(img)
    # plt.show()
    resized_img = resize_and_aspect_crop(img, IMG_RESIZE_SIZE, IMAGE_ASPECT_RATIO)
    env.step()
    return resized_img
    
def camera_execute_actions(env, current_position, trajectory, cur_obs_list, context_size, travers_map, save_state):
    '''
    let camera execute actions according to the given trajectory
    '''
    turn = False
    for subgoal in trajectory:
        sub_waypoints = sub_waypoints_generator(subgoal, current_position)  #generate four sub_waypoints for each next subgoal
        last_sub_waypoint = current_position
        for sub_waypoint in sub_waypoints:
            collision = check_collision(sub_waypoint[:2], travers_map)          
            current_position = last_sub_waypoint
            current_heading_point = sub_waypoint
            resized_img = camera_set_and_record(env, current_position, current_heading_point)
            save_state.append(np.array([current_position[0], current_position[1], current_heading_point[0], current_heading_point[1], collision]))
          
            last_sub_waypoint = sub_waypoint
            time.sleep(0.01)          
            if collision:
                print("collide")               
                break
            
        if len(cur_obs_list) < context_size + 1:
            cur_obs_list.append(resized_img.unsqueeze(0))
        else:
            cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
            
        if collision:
            current_yaw = np.arctan2(current_heading_point[1] - current_position[1], current_heading_point[0] - current_position[0])
            collision_num = 0
            coin = np.random.rand() 
            if coin >= 0.5:
                while collision and collision_num < 8:
                    turn = True
                    # print("turn right")
                    collision_num += 1
                    current_yaw -= 45 / 180 * np.pi
                    current_heading_point[:2] = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
                    
                    resized_img = camera_set_and_record(env, current_position, current_heading_point)
                    save_state.append(np.array([current_position[0], current_position[1], current_heading_point[0], current_heading_point[1], collision]))
                    for i in range(context_size):
                        cur_obs_list.pop(0)
                        cur_obs_list.append(resized_img.unsqueeze(0))

                    collision = check_collision(current_heading_point[:2], travers_map)
            else:
                while collision and collision_num < 8:
                    turn = True
                    # print("turn left")
                    collision_num += 1
                    current_yaw += 45 / 180 * np.pi
                    current_heading_point[:2] = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
                    
                    resized_img = camera_set_and_record(env, current_position, current_heading_point)
                    save_state.append(np.array([current_position[0], current_position[1], current_heading_point[0], current_heading_point[1], collision]))            
                    for i in range(context_size):
                        cur_obs_list.pop(0)
                        cur_obs_list.append(resized_img.unsqueeze(0))

                    collision = check_collision(current_heading_point[:2], travers_map)
            break
    return current_position, current_heading_point, collision, turn

def camera_follow_traj(env, current_position, trajectory, orientation, cur_obs_list, context_size):
    '''
    directly excute the trajectory
    '''
    for i in range(len(trajectory)):
        env.viewer.initial_pos = trajectory[i]
        env.viewer.initial_view_direction = orientation[i] - trajectory[i]
        env.viewer.reset_viewer()
        
        frame = env.renderer.render(modes=("rgb"))
        img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
        resized_img = resize_and_aspect_crop(img, (96,96), 4 / 3)
        if len(cur_obs_list) < context_size + 1:
            cur_obs_list.append(resized_img.unsqueeze(0))
        else:
            cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
        
        env.step()
        time.sleep(0.01)
    return trajectory[-1], orientation[-1]

def sub_waypoints_generator(subgoal, current_position):
    '''
    generate four sub_waypoints for each next subgoal
    '''
    sub_waypoints = []
    for i in range(5):
        sub_waypoints.append(current_position + (subgoal - current_position) * (i + 1) / 5)
    # sub_waypoints.append(subgoal)
    return sub_waypoints


if __name__ == "__main__":
    # load config file
    torch.multiprocessing.set_start_method("spawn")
    config_path = '/path/to/flodiff/test.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # set random seed and some config supp
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    
    # set device
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # set transform
    cudnn.benchmark = True 
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)
        
    # Create the model
    vision_encoder = flona_ViNT(
        context_size=config["context_size"],
        obs_encoding_size=config["encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = flona(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Model load 
    load_flona_path = config["checkpoint_path"]
    latest_flona_checkpoint = torch.load(load_flona_path, map_location ='cpu')
    load_model(model, latest_flona_checkpoint)
    model = model.to(device)

    # Simulator
    headless=False
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    
    # Parameters
    short_exec=True
    metric_waipoint_spacing = 0.045
    waypoint_spacing = 1
    max_steps = -1 if not short_exec else 90
    step = 0
    context_size = config["context_size"] 
    arrive_th = 0.3
    state_save_dir = config["state_save_dir"]
    if not os.path.exists(state_save_dir):
        os.makedirs(state_save_dir)
    img_save_dir = os.path.join(state_save_dir, 'image')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    state_save_dir = os.path.join(state_save_dir, 'trajectory')
    if not os.path.exists(state_save_dir):
        os.makedirs(state_save_dir)
    
    for scene_name_floor in config["test_scenes"]:
        print("testing scene: ", scene_name_floor)  
        scene_id = scene_name_floor.split('_')[0]
        floor = int(scene_name_floor.split('_')[1])
        if floor == 0:
            foucused_map_path =  os.path.join(config["trav_maps_path"], scene_id, 'foucused_map.png')
        else:
            foucused_map_path =  os.path.join(config["trav_maps_path"], scene, 'foucused_map.png')

        
        # load scene
        s = Simulator(
            mode="gui_interactive" if not headless else "headless",
            image_width=512,
            image_height=512,
            rendering_settings=settings,
            device_idx=0
        )
        scene = StaticIndoorScene(
            scene_id,
            build_graph=True,
        )
        s.import_scene(scene)
        if not headless:
            s.viewer.initial_pos = config["initial_pos"]
            s.viewer.initial_view_direction = config["initial_view_direction"]
            s.viewer.reset_viewer()
        # floorplan and travers map
        if floor == 0:
            floorplan_path = config["trav_maps_path"] + scene_id + '/map.png'
            travers_path = config["trav_maps_path"] + scene_id + '/floor_trav_test_0_modified_8bit.png'
        else:
            floorplan_path = config["trav_maps_path"] + scene_name_floor + '/map.png'
            travers_path = config["trav_maps_path"] + scene_name_floor + '/floor_trav_test_' + str(floor) + '_modified_8bit.png'
        floorplan = img_path_to_data(floorplan_path, (96,96))
        if floorplan.shape[0] > 3:
            floorplan = floorplan[:3]
        floorplan = floorplan.unsqueeze(0) # 1,h,w,3
        floorplan_ary = np.array(Image.open(floorplan_path).convert('RGBA') ) # original size
        travers_map = np.array(Image.open(travers_path)) # original size
        # load traj
        trajs_dir = config["testdataset"] + scene_id + '_' + str(floor)
        trajs_id = []
        for f in os.listdir(trajs_dir):
            if f.startswith('traj'):
                trajs_id.append(int(f.split('_')[1]))
        trajs_id.sort()
        
        # iterate trajs
        for i in range(10):
            traj_name = 'traj_' + str(trajs_id[i])
            traj_file = os.path.join(config["testdataset"], f'{scene_id}_{floor}', traj_name, traj_name + '.npy')   
            scene_dir = config["scene_path"] + scene_id
            with open(os.path.join(scene_dir, "floors.txt"), "r") as ff:  # floor z coordinate
                heights = sorted(list(map(float, ff.readlines())))
            base_height = heights[floor]
            height = 0.85 + base_height       
            all_traj = np.load(traj_file)
            start_time = 4
            goal_time = -1
            # start to excute a first few positions--collect context
            trajectory_to_execute = []
            cur_obs_list = []
            trajectory_to_execute = [[all_traj[i][0], all_traj[i][1], height] for i in range(start_time - context_size, start_time + 1)]
            heading_to_execute = [[all_traj[i][2], all_traj[i][3], height] for i in range(start_time - context_size, start_time + 1)]
            for i in range(len(trajectory_to_execute)):
                trajectory_to_execute[i] = np.array(trajectory_to_execute[i])
                heading_to_execute[i] = np.array(heading_to_execute[i])
            goal_pos = np.array([[all_traj[goal_time][0], all_traj[goal_time][1]]])
            current_position, current_heading_point = camera_follow_traj(s, trajectory_to_execute[0], trajectory_to_execute, heading_to_execute, cur_obs_list, context_size)         
            save_num = 0
            save_state = []
            arrive = False
            turn = False
            collision_num = 0
            step = 1
            floorplan_ary = np.array(Image.open(floorplan_path).convert('RGBA') )
            traj_img_save_dir = os.path.join(img_save_dir, scene_name_floor + '_' + traj_name)
            if not os.path.exists(traj_img_save_dir):
                os.makedirs(traj_img_save_dir)
            while step <= max_steps:
                with Profiler("Simulator step"):
                    if len(trajectory_to_execute) > 0:
                        current_position, current_heading_point, collision, turn = camera_execute_actions(s, current_position, trajectory_to_execute, cur_obs_list, context_size, travers_map, save_state)
                        trajectory_to_execute = []
                    if collision:
                        print('into a stuck situation')
                        break

                    if len(cur_obs_list) == context_size + 1:
                        cur_obs = torch.cat(cur_obs_list, dim=0)
                        cur_pos = np.array([current_position[:2]])
                        cur_heading = np.array([current_heading_point[:2]])
                        cur_heading = cur_pos + (cur_heading - cur_pos) / np.linalg.norm(cur_heading - cur_pos)
                        if np.linalg.norm(cur_pos - goal_pos) < arrive_th:
                            print("arrive at the target!")
                            arrive = True
                            break
                       
                        actions_2d = execute_model(model, 
                                                   cur_pos, 
                                                   cur_heading, 
                                                   goal_pos, 
                                                   cur_obs, 
                                                   floorplan, 
                                                   metric_waipoint_spacing, 
                                                   waypoint_spacing, 
                                                   transform, 
                                                   device, 
                                                   noise_scheduler, 
                                                   floorplan_ary, 
                                                   os.path.join(traj_img_save_dir ,f'{save_num}.png')) 
                        save_num += 1
                        for i in range(20):
                            action = actions_2d[i]
                            trajectory_to_execute.append(np.array([action[0], action[1], height]))
                    else:
                        frame = s.renderer.render(modes=("rgb"))
                        img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
                        resized_img = resize_and_aspect_crop(img, IMG_RESIZE_SIZE, IMAGE_ASPECT_RATIO)
                        cur_obs_list.append(resized_img.unsqueeze(0))
                    step += 1
            save_state = np.array(save_state)
            state_save_file = os.path.join(state_save_dir, scene_name_floor + '_' + traj_name + '.txt')
            np.savetxt(state_save_file, save_state, fmt='%f')
        s.disconnect()