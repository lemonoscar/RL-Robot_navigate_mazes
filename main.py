from QRobot import QRobot
from Maze import Maze
from Runner import Runner
import random
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot
from ReplayDataSet import ReplayDataSet
import heapq

def my_search(maze):
    """
    使用 A* 算法在迷宫中寻找从起点到终点的最短路径。

    :param maze: 提供如下接口的迷宫对象：
        - maze.sense_robot(): 获取起点坐标 (row, col)
        - maze.destination: 获取终点坐标 (row, col)
        - maze.can_move_actions(pos): 返回从当前坐标可移动的方向列表，例如 ['u', 'r']
    
    :return: 路径列表，例如 ['u', 'u', 'r']
    """
    direction_offsets = {
        'u': (-1, 0),
        'r': (0, 1),
        'd': (1, 0),
        'l': (0, -1),
    }

    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    start = maze.sense_robot()
    goal = maze.destination
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, []))

    visited = {}
    while open_list:
        f, g, current_pos, path_so_far = heapq.heappop(open_list)
        if current_pos in visited and visited[current_pos] <= g:
            continue
        visited[current_pos] = g
        if current_pos == goal:
            return path_so_far
        for action in maze.can_move_actions(current_pos):
            delta_row, delta_col = direction_offsets[action]
            next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
            new_g = g + 1
            new_f = new_g + heuristic(next_pos, goal)
            heapq.heappush(open_list, (new_f, new_g, next_pos, path_so_far + [action]))

    return []



class Robot(TorchRobot):
    valid_action = ['u', 'r', 'd', 'l']

    def __init__(self, maze):
        """
        初始化 Robot 实例，配置奖励函数，建立观察视角，并执行训练过程。
        """
        super(Robot, self).__init__(maze)
        self.maze = maze

        # 设置奖励策略
        self._configure_reward()

        # 构建完整视野的记忆（用于训练）
        self.memory.build_full_view(maze=self.maze)

        # 开始训练过程
        self.lost_list = self._train_until_success()

    def _configure_reward(self):
        """
        配置迷宫中的奖励函数，用于训练时反馈。
        """
        self.maze.set_reward(reward={
            "hit_wall": 10.0,
            "destination": -70.0,
            "default": 1.0,
        })

    def _train_until_success(self):
        """
        进行训练直到机器人能够成功到达目标。
        :return: 每轮训练损失列表
        """
        lost_list = []
        batch_size = len(self.memory)

        while True:
            # 使用完整内存进行训练更新
            loss = self._learn(batch=batch_size)
            lost_list.append(loss)

            # 训练后进行一次测试以验证是否成功
            if self._test_robot_success():
                break

        return lost_list

    def _test_robot_success(self):
        """
        测试当前模型是否可以从起点成功走到终点。
        :return: 如果成功到达目标，返回 True
        """
        self.reset()
        max_steps = self.maze.maze_size ** 2 - 1

        for _ in range(max_steps):
            _, reward = self.test_update()
            if reward == self.maze.reward["destination"]:
                return True

        return False

    def train_update(self):
        """
        单步训练更新：感知当前状态，选择动作，执行并记录奖励。
        :return: 执行动作和收到的奖励
        """
        # 获取当前状态
        state = self.sense_state()

        # 根据策略选择动作
        action = self._choose_action(state)

        # 执行动作并获取奖励
        reward = self.maze.move_robot(action)

        return action, reward

    def test_update(self):
        """
        测试模式下进行一步决策，不进行学习。
        :return: 执行动作和收到的奖励
        """
        # 将状态转换为 tensor
        state_array = np.array(self.sense_state(), dtype=np.int16)
        state_tensor = torch.tensor(state_array, dtype=torch.float32, device=self.device)

        # 设置为推理模式
        self.eval_model.eval()
        with torch.no_grad():
            q_values = self.eval_model(state_tensor).cpu().numpy()

        # 选择最小 Q 值对应的动作（因为奖励越小越好）
        best_action_index = np.argmin(q_values).item()
        action = self.valid_action[best_action_index]

        # 执行动作并获取奖励
        reward = self.maze.move_robot(action)

        return action, reward



epoch = 25  
maze_size = 10  
training_per_epoch=int(maze_size * maze_size * 1.5)

g = Maze(maze_size=maze_size)
r = Robot(g)
runner = Runner(r)
runner.run_training(epoch, training_per_epoch)

