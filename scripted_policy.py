import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from piper_ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # Calculate the true center of the box for grasping
        # The box_xyz from env_state is the box center position
        box_center_xyz = box_xyz.copy()
        print(f"Box center from env_state: {box_center_xyz}")
        
        # For grasping, we want to target the exact center of the box
        # The box size is 0.02x0.02x0.02 (from XML), so center should be box_xyz
        
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)  # Even less angle for more vertical approach

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 2.0, 0.0], degrees=30)   # Even less angle for more vertical approach

        # Generate random intermediate and final positions on the table
        # Intermediate position (where right arm places the box) - constrained to table area
        intermediate_xyz = np.array([
            np.random.uniform(-0.25, 0.25),  # x: across table width
            np.random.uniform(0.1, 0.35),    # y: front portion of table
            0.07                             # z: table height (matching box initial position)
        ])
        
        # Final position (where left arm places the box) - combined target position
        final_xyz = np.array([
            np.random.uniform(-0.25, 0.25),  # x: across table width
            np.random.uniform(-0.1, 0.25),   # y: back to center portion of table
            0.07                             # z: table height (matching box initial position)
        ])
        
        print(f"Episode positions: box_center={box_center_xyz}, intermediate={intermediate_xyz}, final={final_xyz}")
        
        print(f"Box initial position: {box_xyz}")
        print(f"Box center for grasping: {box_center_xyz}")
        print(f"Intermediate position (right arm target): {intermediate_xyz}")
        print(f"Final position (left arm target): {final_xyz}")

        # Left arm trajectory: waits, then picks up from intermediate position and places at final position
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # sleep with open gripper
            {"t": 290, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # wait for right arm to complete
            {"t": 310, "xyz": intermediate_xyz + np.array([-0.12, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach from further side - much lower
            {"t": 330, "xyz": intermediate_xyz + np.array([-0.06, 0, 0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # get closer - lower
            {"t": 350, "xyz": intermediate_xyz + np.array([-0.02, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # position near center - much lower
            {"t": 365, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # center position
            {"t": 370, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.9}, # slightly below center, start closing
            {"t": 375, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.7}, # continue closing
            {"t": 380, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.5}, # more closed
            {"t": 385, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.3}, # almost closed
            {"t": 390, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.1}, # very closed
            {"t": 395, "xyz": intermediate_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # fully close
            {"t": 400, "xyz": intermediate_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # lift box - lower height
            {"t": 410, "xyz": final_xyz + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # transport to final - lower height
            {"t": 420, "xyz": final_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # lower to table level
            {"t": 430, "xyz": final_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # place fully on table
            {"t": 440, "xyz": final_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.5}, # start opening gripper
            {"t": 450, "xyz": final_xyz + np.array([0, 0, -0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # fully open gripper
            {"t": 460, "xyz": final_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # lift up slightly
            {"t": 470, "xyz": final_xyz + np.array([-0.12, 0, 0.06]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # move away
            {"t": 480, "xyz": final_xyz + np.array([-0.12, 0, 0.06]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # stay away
        ]

        # Right arm trajectory: picks up box and places at intermediate position, then moves away
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep with open gripper
            {"t": 40, "xyz": box_center_xyz + np.array([0.08, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach from side
            {"t": 70, "xyz": box_center_xyz + np.array([0.04, 0, 0.04]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # get closer to box
            {"t": 90, "xyz": box_center_xyz + np.array([0.01, 0, 0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # position near box center
            {"t": 110, "xyz": box_center_xyz + np.array([0, 0, -0.006]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # slightly below center
            {"t": 130, "xyz": box_center_xyz + np.array([0, 0, -0.006]), "quat": gripper_pick_quat_right.elements, "gripper": 0.8}, # start closing
            {"t": 140, "xyz": box_center_xyz + np.array([0, 0, -0.006]), "quat": gripper_pick_quat_right.elements, "gripper": 0.6}, # continue closing
            {"t": 150, "xyz": box_center_xyz + np.array([0, 0, -0.006]), "quat": gripper_pick_quat_right.elements, "gripper": 0.4}, # more closed
            {"t": 160, "xyz": box_center_xyz + np.array([0, 0, -0.006]), "quat": gripper_pick_quat_right.elements, "gripper": 0.2}, # almost closed
            {"t": 170, "xyz": box_center_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # fully closed
            {"t": 180, "xyz": box_center_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # hold grip
            {"t": 190, "xyz": box_center_xyz + np.array([0, 0, 0.06]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # lift box
            {"t": 220, "xyz": intermediate_xyz + np.array([0, 0, 0.06]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # transport to intermediate
            {"t": 240, "xyz": intermediate_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # lower to table level
            {"t": 250, "xyz": intermediate_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # place fully on table
            {"t": 260, "xyz": intermediate_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat_right.elements, "gripper": 0.5}, # start opening gripper
            {"t": 270, "xyz": intermediate_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # fully open gripper
            {"t": 280, "xyz": intermediate_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # lift up slightly
            {"t": 290, "xyz": intermediate_xyz + np.array([0.12, 0, 0.06]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # move away
            {"t": 480, "xyz": intermediate_xyz + np.array([0.12, 0, 0.06]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # stay away
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    print(f"Starting test_policy with task: {task_name}")
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(1, 6):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")
        
        # Add delay between episodes to see results
        # import time
        # time.sleep(3)  # 3 second delay between episodes


if __name__ == '__main__':
    print("=== STARTING SIMULATION ===")
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)
    print("=== SIMULATION COMPLETE ===")

