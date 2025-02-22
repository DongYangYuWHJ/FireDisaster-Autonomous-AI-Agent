import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import distance

def generate_single_ignition(forest_size=50, center_bias=False):
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

def generate_multiple_ignitions(forest_size=50, 
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
# 参数配置 (可自定义)
# ======================
FOREST_SIZE = 50          # 森林尺寸 (N x N)
TREE_DENSITY = 0.8        # 初始树木密度
FIRE_SPREAD_PROB = 0.3    # 单步蔓延概率
BURN_DURATION = 3         # 单棵树燃烧持续时间 (时间步)
IGNITION_POINTS = generate_multiple_ignitions()  # 初始着火点坐标列表 (可设置多个)

# ======================
# 高级状态系统
# ======================
# 状态矩阵：每个元素存储 [当前状态, 剩余燃烧时间]
# 状态编码：
# 0 = 空地
# 1 = 未燃烧树木
# 2 = 燃烧中
# 3 = 已烧毁
forest = np.zeros((FOREST_SIZE, FOREST_SIZE, 2), dtype=int)

# 初始化森林 (状态层)
forest[:,:,0] = np.random.choice(
    [0, 1], 
    size=(FOREST_SIZE, FOREST_SIZE),
    p=[1-TREE_DENSITY, TREE_DENSITY]
)

# 设置初始着火点 (状态层 + 燃烧时间)
for x, y in IGNITION_POINTS:
    forest[x, y, 0] = 2
    forest[x, y, 1] = BURN_DURATION

# ======================
# 单步状态更新函数
# ======================
def update_fire_step():
    new_forest = forest.copy()
    moore_neighbors = [(-1,-1), (-1,0), (-1,1),
                       (0,-1),          (0,1),
                       (1,-1),  (1,0), (1,1)]
    
    for i in range(FOREST_SIZE):
        for j in range(FOREST_SIZE):
            current_state = forest[i,j,0]
            remaining_time = forest[i,j,1]
            
            # 燃烧状态处理
            if current_state == 2:
                if remaining_time > 1:
                    new_forest[i,j,1] -= 1  # 减少剩余时间
                else:
                    new_forest[i,j,0] = 3  # 燃烧结束变为灰烬
                    new_forest[i,j,1] = 0
                    
                # 尝试引燃邻居
                for dx, dy in moore_neighbors:
                    ni, nj = i+dx, j+dy
                    if 0 <= ni < FOREST_SIZE and 0 <= nj < FOREST_SIZE:
                        if (forest[ni,nj,0] == 1) and (np.random.rand() < FIRE_SPREAD_PROB):
                            new_forest[ni,nj,0] = 2
                            new_forest[ni,nj,1] = BURN_DURATION
    
    forest[:] = new_forest[:]

# ======================
# 生成指定时刻的火灾环境
# ======================
def generate_fire_environment(target_step):
    for _ in range(target_step):
        update_fire_step()
    return forest[:,:,0].copy()  # 返回状态层矩阵

# ======================
# 可视化与输出示例
# ======================
if __name__ == "__main__":
    # 生成第10步的火灾环境
    target_matrix = generate_fire_environment(10)
    
    # 可视化
    cmap = colors.ListedColormap(['white','green','red','black'])
    norm = colors.BoundaryNorm([0,1,2,3,4], cmap.N)
    
    plt.figure(figsize=(10,10))
    plt.imshow(target_matrix, cmap=cmap, norm=norm, interpolation='nearest')
    plt.axis('off')
    plt.title(f"Fire Environment at Step 10\nBurn Duration={BURN_DURATION}, Spread Prob={FIRE_SPREAD_PROB}")
    plt.show()
    
    # 打印矩阵示例
    print("环境矩阵示例 (10x10 区域):")
    print(target_matrix[20:30, 20:30])

