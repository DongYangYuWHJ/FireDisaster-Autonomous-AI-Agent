import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import distance

# ======================
# 参数配置 (可自定义)
# ======================
FOREST_SIZE = 20
TREE_DENSITY = 0.7
BURN_DURATION = 5
STEPS = 7
HUMIDITY_BASE = 30

ENV_FACTORS = {
    'humidity': {'weight': -0.015, 'current': 25}, 
    # 'wind': {'speed': 25, 'direction': 'E'}  # 移除风力参数
}

# ======================
# 火点生成函数
# ======================
def generate_single_ignition(forest_size=FOREST_SIZE, center_bias=True):
    """
    生成符合现实火灾特征的单一起火点
    Parameters:
        forest_size (int): 森林区域边长
        center_bias (bool): 是否倾向于中心区域起火（模拟雷击特征）
    Returns:
        list: 包含单个起火点坐标的列表 [(x,y)]
    """
    if center_bias:
        # 中心区域定义为整个区域的30%核心区
        core_size = int(forest_size * 0.3)
        start = (forest_size - core_size) // 2
        end = start + core_size
        x = np.random.randint(start, end)
        y = np.random.randint(start, end)
    else:
        # 全区域随机分布（模拟人为随机起火）
        x = np.random.randint(0, forest_size)
        y = np.random.randint(0, forest_size)
    
    return [(x, y)]

def generate_multiple_ignitions(forest_size=FOREST_SIZE, 
                              num_fires=3,
                              min_distance=15,
                              cluster_ratio=0,
                              edge_bias=False):
    """
    生成符合现实火灾特征的多点起火配置
    Parameters:
        forest_size (int): 森林区域边长
        num_fires (int): 目标火点数量
        min_distance (int): 火点间最小距离（像素单位）
        cluster_ratio (float): 0-1, 集群火点比例（0.7表示70%火点聚集）
        edge_bias (bool): 是否倾向于边缘区域起火（模拟人为纵火特征）
    Returns:
        list: 包含多个起火点坐标的列表 [(x1,y1), (x2,y2),...]
    """
    points = []
    attempts = 0
    max_attempts = 1000
    
    # 生成集群火点
    if cluster_ratio > 0:
        cluster_size = int(num_fires * cluster_ratio)
        # 生成第一个集群中心
        if edge_bias:
            # 边缘区域定义：距离边界<15%区域宽度
            edge_zone = int(forest_size * 0.15)
            cx = np.random.choice([np.random.randint(0, edge_zone), 
                                 np.random.randint(forest_size-edge_zone, forest_size)])
            cy = np.random.choice([np.random.randint(0, edge_zone), 
                                 np.random.randint(forest_size-edge_zone, forest_size)])
        else:
            cx = np.random.randint(0, forest_size)
            cy = np.random.randint(0, forest_size)
        
        # 生成集群内火点
        cluster_points = []
        while len(cluster_points) < cluster_size and attempts < max_attempts:
            x = np.clip(cx + np.random.randint(-5,6), 0, forest_size-1)
            y = np.clip(cy + np.random.randint(-5,6), 0, forest_size-1)
            if all(distance.euclidean((x,y), p) >= min_distance for p in cluster_points):
                cluster_points.append((x,y))
            attempts += 1
        points.extend(cluster_points)
    
    # 生成分散火点
    remaining = num_fires - len(points)
    while remaining > 0 and attempts < max_attempts:
        if edge_bias:
            # 边缘区域采样
            if np.random.rand() < 0.8:
                x = np.random.choice([np.random.randint(0, int(forest_size*0.2)), 
                                    np.random.randint(int(forest_size*0.8), forest_size)])
                y = np.random.randint(0, forest_size)
            else:
                y = np.random.choice([np.random.randint(0, int(forest_size*0.2)), 
                                    np.random.randint(int(forest_size*0.8), forest_size)])
                x = np.random.randint(0, forest_size)
        else:
            x = np.random.randint(0, forest_size)
            y = np.random.randint(0, forest_size)
        
        # 验证最小距离
        if all(distance.euclidean((x,y), p) >= min_distance for p in points):
            points.append((x,y))
            remaining -= 1
        attempts += 1
    
    return points[:num_fires]  # 返回不超过目标数量的有效点

# ======================
# 环境增强的状态系统
# ======================
class EnhancedFireSystem:
    def __init__(self):
        # 初始化森林状态
        self.forest = np.zeros((FOREST_SIZE, FOREST_SIZE, 2), dtype=int)
        self.forest[:,:,0] = np.random.choice(
            [0, 1], 
            size=(FOREST_SIZE, FOREST_SIZE),
            p=[1-TREE_DENSITY, TREE_DENSITY]
        )
        
        # 设置初始火点
        self.ignition_points = generate_multiple_ignitions()
        print(self.ignition_points)
        for x, y in self.ignition_points:
            self.forest[x, y, 0] = 2
            self.forest[x, y, 1] = BURN_DURATION

    def _get_spread_prob(self):
        """计算综合蔓延概率"""
        base = 0.3
        humidity_effect = 1 + (ENV_FACTORS['humidity']['current']-HUMIDITY_BASE)*ENV_FACTORS['humidity']['weight']
        return base * humidity_effect

    def update_step(self):
        new_forest = self.forest.copy()
        spread_prob = self._get_spread_prob()
        moore_neighbors = [(-1,-1), (-1,0), (-1,1),
                          (0,-1),          (0,1),
                          (1,-1),  (1,0), (1,1)]

        for i in range(FOREST_SIZE):
            for j in range(FOREST_SIZE):
                current_state = self.forest[i,j,0]
                remaining_time = self.forest[i,j,1]
                
                if current_state == 2:  # 燃烧中
                    # 更新时间状态
                    if remaining_time > 1:
                        new_forest[i,j,1] -= 1
                    else:
                        new_forest[i,j,0] = 3
                        new_forest[i,j,1] = 0
                    
                    # 传播火势
                    for dx, dy in moore_neighbors:
                        ni, nj = i+dx, j+dy
                        if 0<=ni<FOREST_SIZE and 0<=nj<FOREST_SIZE:
                            if (self.forest[ni,nj,0] == 1) and (np.random.rand() < spread_prob):
                                new_forest[ni,nj,0] = 2
                                new_forest[ni,nj,1] = BURN_DURATION

        self.forest = new_forest

# ======================
# 可视化与执行
# ======================
def visualize(matrix):
    cmap = colors.ListedColormap(['white','green','red','black'])
    norm = colors.BoundaryNorm([0,1,2,3,4], cmap.N)

    plt.figure(figsize=(10,10))
    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

    plt.title(
        f"humidity: {ENV_FACTORS['humidity']['current']}%"
    )
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    system = EnhancedFireSystem()
    for _ in range(STEPS):
        system.update_step()
    visualize(system.forest[:,:,0])
    
