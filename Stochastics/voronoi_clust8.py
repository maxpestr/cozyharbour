import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
from math import sqrt
import random
import shutil
import time

import copy
import numpy as np
from scipy.spatial import Delaunay
import os


N_MAX = 9999
# строит триангуляцию делоне
def build_delaunay_triangulation(points):
    coords = np.array([point.coords for point in points])
    triangulation = Delaunay(coords)
    all_delonay_pairs = []
    all_delonay_pairs_indices = []
    for simplex in triangulation.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        all_delonay_pairs.append((p1, p2))
        all_delonay_pairs.append((p2, p3))
        all_delonay_pairs.append((p3, p1))
        all_delonay_pairs_indices.extend([(p1.index[0], p2.index[0]), (p2.index[0], p3.index[0]), (p3.index[0], p1.index[0])])
    
    '''
    # отладочные выводы
    #print('all_del_pairs(idxs)', [(pair[0].index[0],pair[1].index[0]) for pair in all_delonay_pairs]) 
    #print('all_del_pair_idx_str', set(all_delonay_pairs_indices))
    with open('test_outp.txt', 'w') as file:
        file.write('all_del_pairs(idxs)\n')
        for pair in all_delonay_pairs:
            file.write(f"{pair[0].index[0]}, {pair[1].index[0]}\n")
        file.write('all_del_pair_idx_str\n')
        file.write(str(set(all_delonay_pairs_indices)))
    
    '''
    return triangulation, all_delonay_pairs, set(all_delonay_pairs_indices)

# все граничные точки ищет
def find_boundary_points(all_delonay_idx, points, idx_map):
    boundary_points = []
    for pair in all_delonay_idx:
        idx1, idx2 = pair
        p1 = points[idx_map[idx1]]
        p2 = points[idx_map[idx2]]
        if p1.index[1] != p2.index[1]:  # If points are from different clusters
            if p1 not in boundary_points:
                boundary_points.append(p1)
            if p2 not in boundary_points:
                boundary_points.append(p2)
    return set(boundary_points)

# все граничные разнокластерные по данному id
def find_boundary_points_for_id(point_id, points, idx_map, edge_map):
    pb = edge_map[point_id]
    boundary_points = []
    boundary_points_ids = []
    for pnp in pb:
        if point_id != pnp and points[idx_map[pnp]].index[1] != points[idx_map[point_id]].index[1]:
            boundary_points_ids.append(points[idx_map[pnp]].index[0])
            boundary_points.append(points[idx_map[pnp]])
    
    return set(boundary_points), set(boundary_points_ids)   

# все граничные однокластерные по данному id
def find_internal_points_for_id(point_id, points, idx_map, edge_map):
    # Получаем список рёбер для данной точки
    internal_edges = edge_map[point_id]
    internal_points_ids = []
    internal_points = []
    fall_points=[]
    # Проходим по всем рёбрам и проверяем, принадлежат ли они тому же кластеру
    for pnp in internal_edges:
        fall_points.append(points[idx_map[pnp]])
        if point_id != pnp and points[idx_map[pnp]].index[1] == points[idx_map[point_id]].index[1]:
            internal_points_ids.append(pnp)
            internal_points.append(points[idx_map[pnp]])
    # Возвращаем множество внутрикластерных точек и их идентификаторы
    return set(internal_points),set(internal_points_ids), set(fall_points) 

def update_max_values(maxt, t):
    maxt['at_max'] = max(maxt['at_max'], t.additional_term)
    maxt['ot_max'] = max(maxt['ot_max'], t.omega_term)
    maxt['bpnt_max'] = max(maxt['bpnt_max'], t.bpn_term)
    maxt['neart_max'] = max(maxt['neart_max'], t.near_term)
    maxt['fatht_max'] = max(maxt['fatht_max'], t.fath_term)

# Ниже всякая хрень для отрисовки

def get_point_color(cluster_index, n_clusters, max_clust_id):
    # Generate a colormap for n_clusters discrete colors
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, max_clust_id + 1))  # Add +1 to include all clusters
    discrete_cmap = ListedColormap(colors)

    # Now, use the cluster_index to get the color from the discrete colormap
    return discrete_cmap.colors[cluster_index]

def plot_filled_voronoi_with_boundaries(folder_path, nom, vor, points_clustering, n_clusters, max_clust_id, hp, bp=[]):
    fig, ax = plt.subplots()

    # Generate colormap for n_clusters discrete colors
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_clusters)))

    # Plot Voronoi diagram boundaries and filled regions
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=0.5)

    # Fill Voronoi regions and thicken boundary lines between different clusters
    for point_idx, region in enumerate(vor.point_region):
        polygons = vor.regions[region]
        if all(v >= 0 for v in polygons):  # Check the region vertices are valid (not -1, which indicates an open region)
            polygon = [vor.vertices[i] for i in polygons]
            region_color = get_point_color(points_clustering[point_idx], n_clusters,max_clust_id)
            plt.fill(*zip(*polygon), alpha=0.5, color=region_color)

    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if all(v != -1 for v in ridge_vertices):
            v1, v2 = ridge_vertices
            p1, p2 = ridge_points
            if points_clustering[p1] != points_clustering[p2]:
                plt.plot(vor.vertices[[v1, v2], 0], vor.vertices[[v1, v2], 1], 'k-', linewidth=2)

    clr = 'blue'
    # Mark the original points
    bpc = [tuple(b.coords) for b in bp]
    hpc = [tuple(b.coords) for b in hp]
    for point_idx, point in enumerate(vor.points):
        clr = get_point_color(points_clustering[point_idx], n_clusters, max_clust_id)
        if tuple(point) not in bpc:
            marker = 'o'  # Circle marker for non-boundary points
            size = 20  # Default size for non-boundary points
        else:
            marker = 'D'  # Diamond marker for boundary points
            size = 40  # Larger size for boundary points
        if tuple(point) in hpc:
            marker = '*'  # Circle marker for non-boundary points
            size = max(size,30)
        plt.scatter(point[0], point[1], color=clr, edgecolor='black', marker=marker, s=size, zorder=2)
    
    filtered_points = np.array([ vor.points[i] for i in range(len(vor.points)) if points_clustering[i] !=max_clust_id])
    min_x = filtered_points[:, 0].min()
    max_x = filtered_points[:, 0].max()
    min_y = filtered_points[:, 1].min()
    max_y = filtered_points[:, 1].max()

    # Set the x-axis and y-axis limits based on the minimum and maximum values
    plt.xlim([min_x-1,max_x+1])
    plt.ylim([min_y-1,max_y+1])
    plt.savefig(os.path.join(folder_path, nom))
    plt.close()
    #plt.show()

def visualize_clusters2d(table,folder_path='kartintki', nom=f'0'):

    clusters = {clust_id: [table.points[table.rev_idx_map[i]].coords for i in indices] for clust_id, indices in table.clusters.items()}
    #clusters = copy.deepcopy(table.clusters)
    bp = copy.deepcopy(table.all_boundary_points)
    hp = copy.deepcopy(table.hull_points)
    points_clustering = []
    clusters1 = copy.deepcopy(clusters)
    max_clust_id = max(clusters.keys())+1

    for clust_id, points in clusters1.items():
        points_clustering.extend([clust_id] * len(points))

    points_clustering.extend([max_clust_id, max_clust_id,max_clust_id ,max_clust_id ])
    clusters1[max_clust_id] = [[[N_MAX, N_MAX], [-N_MAX, N_MAX], [N_MAX, -N_MAX], [-N_MAX, -N_MAX]]]

    # Compute Voronoi diagram for all points
    vor = table.get_vor_for_image()

    # Plot the Voronoi diagram with filled regions and cluster boundaries
    plot_filled_voronoi_with_boundaries(folder_path, nom,vor, points_clustering, len(clusters1), max_clust_id,hp,bp)