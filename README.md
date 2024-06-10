#데이터 로딩
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math
os.chdir('C:/Dropbox/02_ DATA/03_data')
raw = o3d.io.read_point_cloud('pcd_final_voxel_downsampled_0.025_2.pcd')
print(len(raw.points))

# o3d.visualization.draw_geometries([original])

#전처리
outlier = raw

while True:
  plane, indices = outlier.segment_plane(distance_threshold = 0.005, ransac_n = 3, num_iterations = 30000,  probability = 1)

  normal_vector = plane[0:3]
  deg = math.degrees(math.acos(np.dot(normal_vector, [0, 0, 1]) / np.linalg.norm(normal_vector)))

  if 85 <= deg <= 95:
    break

  outlier = outlier.select_by_index(indices, invert=True)
  
n_vector = plane[0:2]
x_vector = [1, 0]

rad = math.acos(np.dot(n_vector, x_vector) / np.linalg.norm(n_vector))
print('rad:', rad)


R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rad))

r_original = raw.rotate(R)
r_original_points = np.asarray(r_original.points)


# Get minimum x,y,z values
min_x, min_y, min_z = r_original.get_min_bound()
print(min_x, min_y, min_z)


original_points = r_original_points - [min_x, min_y, min_z]
original_colors = np.asarray(r_original.colors)

original = o3d.geometry.PointCloud()
original.points = o3d.utility.Vector3dVector(original_points)
original.colors = o3d.utility.Vector3dVector(original_colors)

min_x, min_y, min_z = original.get_min_bound()
print("min", min_x, min_y, min_z)
max_x, max_y, max_z = original.get_max_bound()
print("max", max_x, max_y, max_z)

# Compute AABB
aabb = original.get_axis_aligned_bounding_box()
aabb.color = (0, 0, 0)

print('minimum values of point cloud:',np.min(original_points[:,0]),np.min(original_points[:,1]),np.min(original_points[:,2]))

# o3d.visualization.draw_geometries([original, aabb])

aabb_min_x, aabb_min_y, aabb_min_z = aabb.get_min_bound()
print('minimum values of aabb:', aabb_min_x, aabb_min_y, aabb_min_z)
#(3D coordinate axis+ pcd) visualization

axis_length = 100

x_axis = o3d.geometry.LineSet()
x_axis.points = o3d.utility.Vector3dVector([[0, 0, 0], [axis_length, 0, 0]])
x_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
x_axis.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

y_axis = o3d.geometry.LineSet()
y_axis.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, axis_length, 0]])
y_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
y_axis.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

z_axis = o3d.geometry.LineSet()
z_axis.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, axis_length]])
z_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
z_axis.colors = o3d.utility.Vector3dVector([[0, 0, 1]])

# Visualize the point cloud and axes
#o3d.visualization.draw_geometries([original, aabb, x_axis, y_axis, z_axis])

aabb_x_length = np.abs(aabb.max_bound[0] - aabb.min_bound[0])
aabb_y_length = np.abs(aabb.max_bound[1] - aabb.min_bound[1])
aabb_z_length = np.abs(aabb.max_bound[2] - aabb.min_bound[2])

print('x:', aabb_x_length)
print('y:', aabb_y_length)
print('z:', aabb_z_length)

print(aabb.min_bound[0], aabb.min_bound[1], aabb.min_bound[2])

#평면추출
original_pcd = o3d.geometry.PointCloud()
original_pcd.points = o3d.utility.Vector3dVector(original.points)
original_pcd.colors = o3d.utility.Vector3dVector(original.colors)
#window size -- x,y방향 길이 중 작은값의 30%

shorter_length = min(aabb_x_length, aabb_y_length)
longer_length = max(aabb_x_length, aabb_y_length)
print('smaller_length:', shorter_length)
print('longer_length:', longer_length)

x_window = shorter_length * 0.3
y_window = shorter_length * 0.3

nx = np.ceil(aabb_x_length / x_window)
ny = np.ceil(aabb_y_length / y_window)

print('nx:', nx)
print('ny:', ny)

window_x = aabb_x_length / nx
window_y = aabb_y_length / ny
# x-axis ---hyperparameter: distance threshold= 0.005--> 0.01

plane_models_x = []
plane_coords_x = []
plane_indices_x = []
    
w_range = aabb.min_bound[0]

for i in range(int(nx)):
    w_indices = np.where((w_range <= np.asarray(original_pcd.points)[:, 0]) & (np.asarray(original_pcd.points)[:, 0] < w_range + window_x))

    segmented_pcd = original_pcd.select_by_index(w_indices[0])
    plane_model, inliers = segmented_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10000)

    normal_vector = plane_model[:3]
    angle_x = math.degrees(math.acos(np.dot(normal_vector, [1, 0, 0]) / np.linalg.norm(normal_vector)))
    angle_y = math.degrees(math.acos(np.dot(normal_vector, [0, 1, 0]) / np.linalg.norm(normal_vector)))
    angle_z = math.degrees(math.acos(np.dot(normal_vector, [0, 0, 1]) / np.linalg.norm(normal_vector)))

    if ((85 <= angle_z <= 95) and (angle_x <= angle_y)):
        print('Plane equation: {}x + {}y + {}z + {} = 0'.format(plane_model[0], plane_model[1], plane_model[2], plane_model[3]))
        print(len(inliers))

        plane_models_x.append(plane_model)
        plane_indices_x.append(w_indices[0][inliers])

        coords = np.asarray(original_pcd.points)[w_indices[0][inliers]]
        plane_coords_x.append(coords)

        colors = np.asarray(segmented_pcd.colors)
        plane_color = np.array([0, 0, 1])
        colors[inliers] = plane_color

        segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([segmented_pcd])
        
    w_range += window_x

    # y-axis  ---hyperparameter: distance threshold= 0.005--> 0.01

plane_models_y = []
plane_coords_y = []
plane_indices_y = []
    
w_range = aabb.min_bound[1]

for i in range(int(ny)):
    w_indices = np.where((w_range <= np.asarray(original_pcd.points)[:, 1]) & (np.asarray(original_pcd.points)[:, 1] < w_range + window_y))

    segmented_pcd = original_pcd.select_by_index(w_indices[0])
    plane_model, inliers = segmented_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10000)

    normal_vector = plane_model[:3]
    angle_x = math.degrees(math.acos(np.dot(normal_vector, [1, 0, 0]) / np.linalg.norm(normal_vector)))
    angle_y = math.degrees(math.acos(np.dot(normal_vector, [0, 1, 0]) / np.linalg.norm(normal_vector)))
    angle_z = math.degrees(math.acos(np.dot(normal_vector, [0, 0, 1]) / np.linalg.norm(normal_vector)))

    if ((85 <= angle_z <= 95) and (angle_y <= angle_x)):
        print('Plane equation: {}x + {}y +{}z + {} = 0'.format(plane_model[0], plane_model[1], plane_model[2], plane_model[3]))
        print(len(inliers))

        plane_models_y.append(plane_model)
        plane_indices_y.append(w_indices[0][inliers])

        coords = np.asarray(original_pcd.points)[w_indices[0][inliers]]
        plane_coords_y.append(coords)

        colors = np.asarray(segmented_pcd.colors)
        plane_color = np.array([0, 0, 1])
        colors[inliers] = plane_color

        segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([segmented_pcd])
        
    w_range += window_y

    plane_models_x_dic = {}
plane_models_y_dic = {}

for i in range(len(plane_models_x)):
    plane_models_x_dic[i] = plane_models_x[i]

for j in range(len(plane_models_y)):
    plane_models_y_dic[j] = plane_models_y[j]


print(len(plane_models_x_dic), plane_models_x_dic)
print(len(plane_models_y_dic), plane_models_y_dic)


#외벽모서리 위치 추정
#find orthogonal planes and those intersection point----- 20230725방법(*update: x, y좌표 모두 비교해야함.(230420기존방법이 나을 수도)
orthogonal = []
intersections = []

r = 3.0

for i in range(len(plane_models_x)):
    for j in range(len(plane_models_y)):
        y_cond = (min(plane_coords_x[i][:,1]) - r < np.mean(plane_coords_y[j][:,1])) and (np.mean(plane_coords_y[j][:,1]) < max(plane_coords_x[i][:,1])+ r) 
        x_cond = (abs(min(plane_coords_y[j][:,0]) -  np.mean(plane_coords_x[i][:,0])) < r)  or (abs(np.mean(plane_coords_x[i][:,0]) - max(plane_coords_y[j][:,0])) < r)
        # print(i, j, '|', min(plane_coords_x[i][:,1]), np.mean(plane_coords_y[j][:,1]), max(plane_coords_x[i][:,1]))
        # print(i, j, x_cond,'|', min(plane_coords_y[j][:,0]), np.mean(plane_coords_x[i][:,0]), max(plane_coords_y[j][:,0]))
        # print(y_cond, x_cond) 
    
        if (y_cond and x_cond):
            orthogonal.append((i,j))

            intersection = (np.mean(plane_coords_x[i][:,0]),np.mean(plane_coords_y[j][:,1]))
            intersections.append(intersection)

print(orthogonal)
print('total # of edges:', len(orthogonal))
print(intersections)

intersections_dics = {}
for i in range(len(intersections)):
    intersections_dics[i] = intersections[i]

print(len(intersections_dics), intersections_dics)

#Edge extraction
k_n = 100
thresh = 0.03

pcd_np = np.asarray(original_pcd.points)

kdtree = o3d.geometry.KDTreeFlann(original_pcd)
k_neighbors = [kdtree.search_knn_vector_3d(pcd_np[i], k_n) for i in range(len(pcd_np))]

eigenvalues = []
for neighbors in k_neighbors:
    neighbor_points = pcd_np[neighbors[1]]
    cov_matrix = np.cov(neighbor_points.T)
    eigenvalue, _ = np.linalg.eigh(cov_matrix)
    eigenvalues.append(eigenvalue)

eigenvalues = np.array(eigenvalues)  
e1, e2, e3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
sum_eg = np.add(np.add(e1, e2), e3)
sigma = np.divide(e1, sum_eg)

sigma = sigma > thresh
edge_points = pcd_np[sigma]

edge_pcd = o3d.geometry.PointCloud()
edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
edge_pcd.paint_uniform_color([1,0,0])

print(len(edge_pcd.points))
print(np.asarray(edge_pcd.points))

aabb_edge = edge_pcd.get_axis_aligned_bounding_box()
aabb_edge.color = (0, 0, 0)

# o3d.visualization.draw_geometries([edge_pcd])

# 외벽 모서리 추정
intersection_edges = []

r = 0.2

for i in range(len(intersections)):
    x = intersections[i][0]
    y = intersections[i][1]
    for j in range(len(edge_points)):
        x_cond1 = abs(edge_points[j][0] - x) <= r
        y_cond1 = abs(edge_points[j][1] - y) <= r

        if(x_cond1 and y_cond1):
            intersection_edges.append(edge_points[j])
 


intersection_edges_pcd = o3d.geometry.PointCloud()
intersection_edges_pcd.points = o3d.utility.Vector3dVector(intersection_edges)
intersection_edges_pcd.paint_uniform_color([0,0,1])

# o3d.visualization.draw_geometries([intersection_edges_pcd, edge_pcd, original_pcd])

#클러스터링
from scipy.spatial import KDTree
import numpy as np

def find_neighbors_kdtree(tree, point_idx, eps):
    # Query the pre-built tree for neighbors within eps distance
    neighbors = tree.query_ball_point(tree.data[point_idx], eps)
    return neighbors

def dbscan(points, eps, min_points, print_progress=False):
    # Build the KD-tree once using only x and y coordinates
    tree = KDTree(points[:, :2])

    cluster_label = 0
    cluster_labels = np.full(len(points), -1)  # Initialize labels to -1 (unclassified)
    visited = set()

    for point_idx in range(len(points)):
        if cluster_labels[point_idx] != -1 or point_idx in visited:  # Already classified or visited
            continue

        neighbors = find_neighbors_kdtree(tree, point_idx, eps)
        visited.add(point_idx)

        if len(neighbors) < min_points:
            cluster_labels[point_idx] = -2  # Label as noise
            continue

        # Start a new cluster
        cluster_labels[point_idx] = cluster_label

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if cluster_labels[neighbor_idx] == -2:  # Convert noise point to border point
                cluster_labels[neighbor_idx] = cluster_label

            if cluster_labels[neighbor_idx] == -1:  # Unvisited point
                cluster_labels[neighbor_idx] = cluster_label
                new_neighbors = find_neighbors_kdtree(tree, neighbor_idx, eps)
                visited.add(neighbor_idx)

                if len(new_neighbors) >= min_points:
                    neighbors = list(set(neighbors + new_neighbors))  # Efficiently combine and deduplicate neighbors

            i += 1

        cluster_label += 1  # Move to the next cluster

        if print_progress and point_idx % 100 == 0:
            print(f"Processed point {point_idx + 1}/{len(points)}")

    return cluster_labels

# Example usage
eps = 0.1
min_points = 50
points = np.random.rand(1000, 3)  # Replace with your data
labels = dbscan(points, eps, min_points, print_progress=True)

# Convert the list of points to a NumPy array
# Replace 'intersection_edges' with your list of points
points = np.array(intersection_edges)  # Assuming 'intersection_edges' is your list of points

# Now you can call your dbscan function
eps = 0.1
min_points = 50
labels = dbscan(points, eps, min_points, print_progress=True)

print(labels)
max_label = labels.max()
print('point cloud has {} clusters'.format(max_label + 1))


#cluster된 point에 color입히기---각 cluster마다 색 입혀짐  / tab20: 20가지 색상
#(labels / (max_label if max_label > 0 else 1): label을 0~1사이 값으로 nomalize
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

#cluster되지 않은 points는 black
colors[labels < 0] = 0
intersection_edges_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])    #colors의 값을 3개씩 잘라서 넣어줌

o3d.visualization.draw_geometries([intersection_edges_pcd])
#custom clustering based on Z-Axis distance


# Assume intersection_edges_pcd is your point cloud

# Convert Open3D PointCloud to NumPy array
intersection_edges_np = np.asarray(intersection_edges)

# Sort points by their z-value
intersection_edges_np_sorted = intersection_edges_np[np.argsort(intersection_edges_np[:, 2])]

# Custom clustering based on z-axis distance
xy_thresh = 0.01  # Set your x,y-axis distance threshold
labels = []
cluster_label = 0
for i in range(len(intersection_edges_np_sorted) - 1):
    labels.append(cluster_label)
    x_cond2 = abs(intersection_edges_np_sorted[i][0] - intersection_edges_np_sorted[i + 1][0]) < xy_thresh
    y_cond2 = abs(intersection_edges_np_sorted[i][1] - intersection_edges_np_sorted[i + 1][1]) < xy_thresh
    if (x_cond2 and y_cond2) :
        cluster_label += 1
labels.append(cluster_label)  # For the last point

# Assigning labels back to the original point cloud
sorted_indices = np.argsort(intersection_edges_np[:, 2])
original_labels = np.empty_like(labels)
original_labels[sorted_indices] = labels

# Coloring based on clusters
max_label = max(original_labels)
colors = plt.get_cmap("tab20")(original_labels / (max_label if max_label > 0 else 1))
colors[original_labels < 0] = 0
intersection_edges_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualization
o3d.visualization.draw_geometries([intersection_edges_pcd])

xy_thresh = 0.01  # XY plane distance threshold
minPts = 10        # Minimum number of points for a cluster

def get_neighbors(pt, points, thresh):
    return np.where(np.linalg.norm(points - pt, axis=1) < thresh)[0]

labels = np.full(len(intersection_edges_np), -1)  # Initialize labels to -1 (unclassified)
cluster_label = 0

for i in range(len(intersection_edges_np)):
    if labels[i] != -1:
        continue

    neighbors = get_neighbors(intersection_edges_np[i], intersection_edges_np, xy_thresh)
    if len(neighbors) < minPts:
        labels[i] = -2  # Label as noise
    else:
        labels[i] = cluster_label
        k = 0
        while k < len(neighbors):
            point_index = neighbors[k]
            if labels[point_index] == -2:  # Previously labeled as noise
                labels[point_index] = cluster_label
            elif labels[point_index] == -1:  # Not yet classified
                labels[point_index] = cluster_label
                point_neighbors = get_neighbors(intersection_edges_np[point_index], intersection_edges_np, xy_thresh)
                if len(point_neighbors) >= minPts:
                    neighbors = np.append(neighbors, point_neighbors)
            k += 1
        cluster_label += 1

# Coloring and Visualization
max_label = max(labels)
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Noise points in black
intersection_edges_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([intersection_edges_pcd])

labels = np.asarray(intersection_edges_pcd.cluster_dbscan(eps = 0.2, min_points = 30, print_progress = True))
print(labels)
max_label = labels.max()

print('point cloud has {} clusters'.format(max_label + 1))

#cluster된 point에 color입히기---각 cluster마다 색 입혀짐  / tab20: 20가지 색상
#(labels / (max_label if max_label > 0 else 1): label을 0~1사이 값으로 nomalize
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

#cluster되지 않은 points는 black
colors[labels < 0] = 0
intersection_edges_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])    #colors의 값을 3개씩 잘라서 넣어줌

o3d.visualization.draw_geometries([intersection_edges_pcd])

labels_np = np.asarray(labels)

clusters = []

for i in tqdm(range(max_label + 1)):     #cluster 의 수만큼
    mask = labels_np == i   #boolean type (True, Faluse)  #cluster에 속한 point만 True값
    idx = np.where(mask)[0]   #it returns the indices where the boolean array 'mask' is True.(=cluster에 속한 point의 index만 저장됨.)

    cluster = np.asarray(edge_pcd.points)[idx]
    clusters.append(cluster)

print(len(clusters))
print(clusters)


#수직방향의 edge cluster만 추출
threshold = 0.5
vertical_clusters = []

for i in range(len(clusters)):
    cov = np.cov(clusters[i].T)
    eval, evec = np.linalg.eigh(cov)

    argmax_eval = np.argmax(eval)   #eigenvalue중 가장 큰 값
    argmax_evec = evec[:, argmax_eval]

    if argmax_evec[2] > threshold:
        vertical_clusters.append(i)

print(len(vertical_clusters))
print(vertical_clusters)

vertical_points = []
for j in range(len(vertical_clusters)):
    vertical_points.append(clusters[vertical_clusters[j]])

print(vertical_points)
vertical_idcs = []

edge_pcd_np = np.asarray(edge_pcd.points)

for i in tqdm(range(len(vertical_points))): 
    mask = np.all(np.isin(edge_pcd_np, vertical_points[i]), axis = 1)   #edge_pcd_np와 vertical_pointsdml [i] index에 동시에 있는 것의 index만 추출(True, False)
    idx = np.where(mask)[0]  #True인 것만
    vertical_idcs.append(idx)

print(vertical_idcs)

o3d.visualization.draw_geometries([edge_pcd.select_by_index(np.concatenate(np.asarray(vertical_idcs, dtype = object))), line_set])


# Initialize the dictionary to hold the split point clouds
splits_pcd_dics= {}

# Get all points from the point cloud as a numpy array
original_points_np = np.asarray(original_pcd.points)

# Iterate over the splits dictionary
r = 1
for i in range(len(intersections_dics)):
    # Get the x and y ranges for this split
    x_range = (intersections_dics[i][0] - r, intersections_dics[i][0] + r)    #intersection 반경 r m 범위 안의 포인트들만 선택
    y_range = (intersections_dics[i][1] - r, intersections_dics[i][1] + r)

    # Find points that are within this range
    within_range_indices = np.where(
        (x_range[0] <= original_points_np[:, 0]) & 
        (original_points_np[:, 0] < x_range[1]) & 
        (y_range[0] <= original_points_np[:, 1]) & 
        (original_points_np[:, 1] < y_range[1])
    )[0]

    # Create a new point cloud for these points
    split_pcd = original_pcd.select_by_index(within_range_indices)

    # Store the new point cloud in the dictionary
    splits_pcd_dics[i] = split_pcd


print(splits_pcd_dics)

edge_points_dics = {}
edge_pcd_dics = {}

#define hyperparameters
k_n = 100
thresh = 0.03

for i in range(len(splits_pcd_dics)):

    pcd_np = np.asarray(splits_pcd_dics[i].points)

    #find neighbors
    kdtree = o3d.geometry.KDTreeFlann(splits_pcd_dics[i])
    k_neighbors = [kdtree.search_knn_vector_3d(pcd_np[j], k_n) for j in range(len(pcd_np))]

    eigenvalues = []
    for neighbors in k_neighbors:
        neighbor_points = pcd_np[neighbors[1]]
        cov_matrix = np.cov(neighbor_points.T)
        eigenvalue, _ = np.linalg.eigh(cov_matrix)
        eigenvalues.append(eigenvalue)

    eigenvalues = np.array(eigenvalues)  
    e1, e2, e3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    sum_eg = np.add(np.add(e1, e2), e3)
    sigma = np.divide(e1, sum_eg)

    sigma = sigma > thresh

    edge_points_dics[i] = pcd_np[sigma]

    edge_pcd_dics[i] = o3d.geometry.PointCloud()
    edge_pcd_dics[i].points = o3d.utility.Vector3dVector(edge_points_dics[i])
    edge_pcd_dics[i].paint_uniform_color([1,0,0])


print(len(edge_pcd_dics), edge_pcd_dics)
   
#기울기 계산
edge_dirs = []
tilts = []

for i in edge_clusters.keys():
    label = edge_clusters[i]
    selected_points = points[labels == label]

    covariance = np.cov(selected_points.T)
    evalue, evector = np.linalg.eigh(covariance)

    argmax_evalue = np.argmax(evalue)
    argmax_evector = evector[:, argmax_evalue]
    edge_dirs.append(argmax_evector)
 
    tilt = math.degrees(math.acos(np.dot(argmax_evector, [0, 0, 1]) / np.linalg.norm(argmax_evector)))
    if tilt > 90:
        tilt = 180 - tilt
    tilts.append(tilt)

for i in range(len(tilts)):
    print(i,":",tilts[i])
