import numpy as np
import tqdm
import os

def judge_success(data, distance_th_times, collision_th, suc_dis, shortest_traj):
    # cal distance in shortest_traj
    shortest_dis = 0
    for i in range(len(shortest_traj)-1):
        shortest_dis += np.linalg.norm(shortest_traj[i][:2] - shortest_traj[i+1][:2])
 
    goal = shortest_traj[-1][:2]
    d_0 = np.linalg.norm(data[0][:2] - goal)
    collision_num = 0
    arrive = False
    cul_dis = 0   
    for i,d in enumerate(data):
        d_t = np.linalg.norm(d[:2] - goal)
        if i > 0:
            cul_dis += np.linalg.norm(d[:2] - data[i-1][:2])
        if collision_num >= collision_th:
            break
        if d[4] == 1 :
            collision_num += 1
        if np.linalg.norm(d[:2] - goal) < suc_dis:
            arrive = True
            break
    return arrive, collision_num, shortest_dis, cul_dis, d_0, d_t

traj_dir = '/path/to/flodiff/dataset/scenes_117/test/'
distance_th_times = 3
collision_th = [1, 10, 30, 50, 5000]
suc_dis = [0.25, 0.3, 0.35, 0.4]

for c_th in collision_th:
    for s_dis in suc_dis:
        print('-------------------------------------')
        print('collision count th:', c_th, 'arrive th:', s_dis)

        data_dir = os.path.join('/path/to/flodiff/results/exp_1')
        arrives = []
        collision_nums = []
        shortest_diss = []
        cul_diss = []
        SPL = []
        SoftSPL = []
        # loss_rate = 0
        for f in os.listdir(data_dir):
            # get the shortest trajectory  
            f_splits = f.split('_')
            f_splits[-1] = f_splits[-1].split('.')[0]
            scene_floor = f_splits[0] + '_' + f_splits[1]
            traj_name = f_splits[2] + '_' + f_splits[3]
            shortest_traj_file = os.path.join(traj_dir, scene_floor, traj_name, traj_name+'.npy')
            shortest_traj_file = np.load(shortest_traj_file)
            data = np.loadtxt(os.path.join(data_dir, f)) 
            arrive, collision_num, shortest_dis, cul_dis, d_0, d_t= judge_success(data, distance_th_times, c_th, s_dis, shortest_traj_file)
            arrives.append(arrive)
            collision_nums.append(collision_num)
            shortest_diss.append(shortest_dis)
            cul_diss.append(cul_dis)
            SPL.append(arrive * shortest_dis / max(cul_dis, shortest_dis))
            SoftSPL.append((1 - d_t / d_0) * shortest_dis / max(cul_dis, shortest_dis))
        
        
        print("exp_1:-----SR:", np.mean(arrives), "SPL:", np.mean(SPL), "SoftSPL:", np.mean(SoftSPL))
        print('collision mean nums:', np.mean(collision_nums))
            
            
