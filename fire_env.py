import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import fireSimulator
from matplotlib import colors, patches
matplotlib.use('TkAgg')

class SimpleFireEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 100}

    def __init__(self, grid_size=fireSimulator.FOREST_SIZE, fire_spread_prob=0, render_mode=None,max_steps=50):
        super().__init__()

        self.grid_size = grid_size
        self.fire_spread_prob = fire_spread_prob
        self.render_mode = render_mode
        self.max_steps=max_steps

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size ** 2 + 2,),  # 网格状态(64) + 坐标(2)
            dtype=np.float32
        )

        # 环境状态
        self.grid = None
        self.agent_pos = None
        self.water = 5

        self.fig = None
        self.ax = None
        self.img = None
        self.agent_marker = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.img = None
            self.agent_marker = None

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 环境编辑核心代码：
        # for _ in range(3):
        #     x, y = self.np_random.integers(0, self.grid_size, size=2)
        #     self.grid[x][y] = 1.0
        # self.grid = fireSimulator.finish_all_step_and_return()
        self.envSystem = fireSimulator.init_and_return_new_env()
        self.UIgrid = self.envSystem.forest[:, :, 0]
        self.grid = fireSimulator.whole_forest_in_bits(self.UIgrid)

        self.agent_pos = self.np_random.integers(0, self.grid_size, size=2)
        self.water = 99999999

        if self.render_mode == 'human':
            self._init_render()
        self.current_step=0
        return self._get_obs(), {}
    
    def _grid_update(self):
        self.envSystem.update_step()
        # print("env update")
        self.UIgrid = self.envSystem.forest[:, :, 0]
        self.grid = fireSimulator.whole_forest_in_bits(self.envSystem.forest[:, :, 0])

    #小人的step
    def step(self, action):
        self.extinguished_this_step=False
        self.current_step+=1
        self._grid_update()
        # 执行动作
        if action < 4:
            self._move_agent(action)
        else:
            self._extinguish()

        # 火势蔓延
        # self._spread_fire()

        # 计算奖励
        reward = self._calculate_reward()

        # 终止条件
        terminated = np.sum(self.grid) == 0  # 所有火熄灭
        truncated = self.current_step>=self.max_steps

        if (truncated):
            reward-=10.0
        if terminated:
            reward+=50.0

        # 渲染更新
        if self.render_mode == 'human':
            self._update_render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _move_agent(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        dx, dy = moves[action]
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        self.agent_pos = [new_x, new_y]

    def _extinguish(self):
        if self.water > 0:
            x, y = self.agent_pos
            if self.grid[x][y] == 1.0:
                self.grid[x][y] = 0.0
                self.UIgrid[x][y] = 0.0
                self.water -= 1
                self.extinguished_this_step=True

    # def _spread_fire(self):
    #     new_fire = []
    #     for x in range(self.grid_size):
    #         for y in range(self.grid_size):
    #             if self.grid[x][y] == 1.0:
    #                 # 向四个方向传播
    #                 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #                     nx, ny = x + dx, y + dy
    #                     if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
    #                         if self.grid[nx][ny] == 0.0 and self.np_random.random() < self.fire_spread_prob:
    #                             new_fire.append((nx, ny))
    #     for x, y in new_fire:
    #         self.grid[x][y] = 1.0

    def _calculate_reward(self):
        reward = 0.0
        reward-=0.01
        reward+=10*self.extinguished_this_step
        reward-=0.01*np.sum(self.grid)

        fire_positions = np.argwhere(self.grid == 1.0)
        if len(fire_positions) > 0:
            distances = np.linalg.norm(fire_positions - self.agent_pos, axis=1)
            reward += 0.1 / (np.min(distances) + 1.0)

        return reward

    def _get_obs(self):
        return np.concatenate([
            self.grid.flatten(),
            self.agent_pos
        ]).astype(np.float32)

    # def _init_render(self):
    #     plt.ion()  # 开启交互模式
    #     self.fig, self.ax = plt.subplots()
    #     self.img = self.ax.imshow(self.grid, cmap='hot', vmin=0, vmax=1)
    #     self.agent_marker = self.ax.scatter(
    #         self.agent_pos[1],
    #         self.agent_pos[0],
    #         c='blue', s=100, marker='s', edgecolors='white'
    #     )
    #     plt.title("Simple Fire Environment")
    #     plt.show(block=False)
    #     self.fig.canvas.draw()

    # def _update_render(self):
    #     if self.fig is None:
    #         self._init_render()
    #     else:
    #         self.img.set_data(self.grid)
    #         self.agent_marker.set_offsets([self.agent_pos[1], self.agent_pos[0]])
    #         self.fig.canvas.draw()
    #         self.fig.canvas.flush_events()
    #         plt.pause(1 / self.metadata['render_fps'])

    from matplotlib import colors

    def _init_render(self):
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots()

        # 定义颜色映射：绿色=未燃，红色=着火，黄色=燃烧，黑色=灰烬
        cmap = colors.ListedColormap(['white','green','red','black'])
        bounds = [0, 1, 2, 3, 4]  # 设置边界，确保颜色正确映射
        norm = colors.BoundaryNorm(bounds, cmap.N)

        self.img = self.ax.imshow(self.UIgrid, cmap=cmap, norm=norm, interpolation="nearest")
        
        # Add legend (color explanation)
        legend_patches = [
            patches.Patch(color='white', label="Empty Ground (Non-Flammable)"),
            patches.Patch(color='green', label="Trees (Flammable)"),
            patches.Patch(color='red', label="Burning"),
            patches.Patch(color='black', label="Burned-out (Ashes)")
        ]
        self.ax.legend(handles=legend_patches, loc='upper right')
        
        self.agent_marker = self.ax.scatter(
            self.agent_pos[1],
            self.agent_pos[0],
            c='blue', s=100, marker='s', edgecolors='white'
        )

        plt.title("Simple Fire Environment")
        plt.show(block=False)
        self.fig.canvas.draw()

    def _update_render(self):
        if self.fig is None:
            self._init_render()
        else:
            self.img.set_data(self.UIgrid)
            self.agent_marker.set_offsets([self.agent_pos[1], self.agent_pos[0]])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata['render_fps'])


    def render(self):
        if self.render_mode == 'rgb_array':
            return self.img.get_array()
        return None

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None



