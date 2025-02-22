import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义森林大小
forest_size = 50

# 初始化森林矩阵：0表示空地，1表示树木，-1表示燃烧过的区域
forest = np.random.choice([0, 1], size=(forest_size, forest_size), p=[0.2, 0.8])

# 随机选择一些初始燃烧点
num_initial_fires = 5
initial_fire_indices = np.random.choice(forest_size * forest_size, num_initial_fires, replace=False)
for index in initial_fire_indices:
    x, y = divmod(index, forest_size)
    forest[x, y] = -1

# 定义火灾蔓延的概率和模拟的时间步数
spread_probability = 0.3
time_steps = 10

# 定义邻居相对位置（上下左右）
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 模拟火灾蔓延过程
for _ in range(time_steps):
    new_forest = forest.copy()
    for i in range(forest_size):
        for j in range(forest_size):
            if forest[i, j] == 1:  # 当前是树木
                for dx, dy in neighbors:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < forest_size and 0 <= nj < forest_size and forest[ni, nj] == -1:
                        if np.random.rand() < spread_probability:
                            new_forest[i, j] = -1  # 树木被点燃
                            break
    forest = new_forest

print(forest)

# 可视化最终的森林状态
plt.figure(figsize=(10, 10))
sns.heatmap(forest, cmap='YlGnBu', cbar=False, square=True, linewidths=.5)
plt.title('Forest Fire Simulation After {} Steps'.format(time_steps))
plt.show()
