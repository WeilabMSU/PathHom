""" Summary: python code for the (persistent) path homology.

    Author:f
        Dong Chen
    Create:
        2022-04-12
    Modify:
        2022-06-30
    Dependencies:
        python                    3.7.4
        numpy                     1.21.5
"""


import numpy as np
import copy
import argparse
import sys


class PathHomology(object):
    def __init__(self, initial_axes=None):
        self.initial_axes = initial_axes
        self.initial_vector_x = np.array([1, 0, 0])
        self.initial_vector_y = np.array([0, 1, 0])
        self.initial_vector_z = np.array([0, 0, 1])
        self.save_temp_result = False
        return None

    @staticmethod
    def vector_angle(v0, v1):
        """_summary_ Calculate the angle between vector v0 and vector v1 in degree.

            Args:
                v0 (array): n dimension vector, n >= 2
                v1 (array): n dimension vector, n >= 2

            Returns:
                angle (int): angle in degree.
        """
        v0_u = v0 / np.linalg.norm(v0)
        v1_u = v1 / np.linalg.norm(v1)
        angle = np.degrees(np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0)))
        return angle

    @staticmethod
    def remove_loops(edges):
        """_summary_Remove the loops of the digraph.
            Args:
                edges (array): shape = [n, 2]

            Returns:
                edges (array): shape = [n-m, 2], m is the number of the loops
        """
        loop_idx = []
        loop_nodes = []
        for i, e in enumerate(edges):
            if e[0] == e[1]:
                loop_idx.append(i)
                loop_nodes.append(e[0])
        if len(loop_nodes) > 0:
            print(f'Warning, loops on node {loop_nodes} were removed.')
        edges = np.delete(edges, loop_idx, axis=0)
        return edges

    @staticmethod
    def split_independent_compondent(edges, nodes):
        """_summary_ If the digraph is not fully connected, then splitting it into independent components.
                    Using the depth first search (DFS) algorithms to split the undirected graph.

            Args:
                edges (array): shape = [n, 2]
                nodes (array): shape = [k, ], k is the number of the whole graph

            Returns:
                all_components (list): the nodes' set of independent components
        """
        # convert into str
        node_map_idx = {node: idx for idx, node in enumerate(nodes)}

        # adjacency list of the graph
        graph = [[] for i in range(len(nodes))]
        for i, one_edge in enumerate(edges):
            u, v = one_edge
            # Assuming graph to be undirected.
            graph[node_map_idx[u]].append(v)
            graph[node_map_idx[v]].append(u)

        # components list
        all_components = []
        visited = [False for n in nodes]

        def depth_first_search(node, component):
            # marking node as visited.
            visited[node_map_idx[node]] = True

            # appending node in the component list
            component.append(node)
            # visiting neighbours of the current node
            for neighbour in graph[node_map_idx[node]]:
                # if the node is not visited then we call dfs on that node.
                if visited[node_map_idx[neighbour]] is False:
                    depth_first_search(neighbour, component)
            return None

        for i, one_node in enumerate(nodes):
            if visited[i] is False:
                component = []
                depth_first_search(one_node, component)
                all_components.append(component)

        return all_components

    @staticmethod
    def split_independent_digraph(all_components, edges):
        """_summary_: If the digraph is not fully connected, then splitting it into independent components.
                    Using the depth first search (DFS) algorithms to split the undirected graph.

            Args:
                all_components (list): the nodes' set of independent components
                edges (array): shape = [n, 2]

            Returns:
                all_digraphs (list): a list of digraphs, each digraph contains a list of edges.
        """
        all_digraphs = [[] for i in all_components]
        edges_visited = [False for i in edges]
        for i_c, component in enumerate(all_components):
            for i_e, edge in enumerate(edges):
                if (edges_visited[i_e] is False) and (edge[0] in component or edge[1] in component):
                    all_digraphs[i_c].append(edge)
                    edges_visited[i_e] = True
            if len(component) == 1 and np.shape(all_digraphs[i_c])[0] < 1:
                all_digraphs[i_c].append(component)

        return all_digraphs

    def utils_generate_allowed_paths(self, edges, max_path):
        """_summary_Generate the allowed paths of the digraph.
            Args:
                edges (array): shape = [n, 2], int or str

            Returns:
                allowed_path_str (dict): all paths from dimension 0 to dimension max_path
        """
        # digraph info
        nodes = np.unique(edges)
        nodes_num = len(nodes)
        nodes_idx_map = {node: idx for idx, node in enumerate(nodes)}

        # edges matrix, start->end = row->column
        edge_matrix = np.zeros([nodes_num, nodes_num])
        for i, edge in enumerate(edges):
            edge_matrix[nodes_idx_map[edge[0]], nodes_idx_map[edge[1]]] = 1

        # path_0 = vertex set
        allowed_path = {0: [np.array([n]) for n in nodes]}
        allowed_path_str = {0: [str(n) for n in nodes]}

        # path_(1 to max_path)
        for i in range(0, max_path+1):
            allowed_path[i+1] = []
            allowed_path_str[i+1] = []
            for path_previous in allowed_path[i]:
                for node in nodes:
                    if edge_matrix[nodes_idx_map[path_previous[-1]], nodes_idx_map[node]]:
                        new_path = np.append(path_previous, node)
                        allowed_path[i+1].append(new_path)
                        allowed_path_str[i+1].append('->'.join([str(one_node) for one_node in new_path]))
                    else:
                        continue
        return allowed_path_str

    def utils_unlimited_boundary_operator(self, allowed_path, max_path):
        """_summary_: Generate the n-th boundary matrix for mapping (n+1)-path to (n)-path.

            Args:
                path_n_1 (list): list of (n+1)-paths, each path stores in array
                n (int): n should >= 1, n is the dimension of the boundary matrix

            Returns:
                unlimited_boundary_mat (array): the matrix representation of the n-th boundary matrix
        """
        # For D_0, matrix is [0]*len(nodes)
        boundary_map_matrix = {0: np.zeros([len(allowed_path[0]), ])}
        boundary_mat_matrix_rank = {0: 0}
        allowed_path_idx_argument = {0: [1]*len(allowed_path[0])}

        for n in range(1, max_path+2):
            boundary_map_dict = {}
            boundary_operated_path_name_collect = []

            allowed_path_n_types = len(allowed_path[n])
            if allowed_path_n_types == 0:
                boundary_map_matrix[n] = np.zeros([1, len(allowed_path[n-1])])
                boundary_mat_matrix_rank[n] = 0
                allowed_path_idx_argument[n] = [1] * len(allowed_path[n-1])
                break

            for i_path, path in enumerate(allowed_path[n]):

                # split the path into nodes with idx
                path_node_idx = path.split('->')

                # record the result path after boundary operation
                boundary_operated_path_info = {}
                for i_kill in range(n+1):
                    # kill the  i_kill-th vertex
                    temp_path = np.delete(path_node_idx, i_kill)
                    temp_path_str = '->'.join([str(pp) for pp in temp_path])
                    boundary_operated_path_info[temp_path_str] = (-1)**(i_kill)

                    # record all possible n_path
                    boundary_operated_path_name_collect.append(temp_path_str)
                boundary_map_dict[path] = copy.deepcopy(boundary_operated_path_info)

            # generate the boundary matrix, D; row_p * column_p = n_1_path * n_path
            considered_operated_path_name = np.unique(boundary_operated_path_name_collect + allowed_path[n-1])
            unlimited_boundary_mat = np.zeros([allowed_path_n_types, len(considered_operated_path_name)])
            for i_path, (n_1_path_str, operated_n_path_dict) in enumerate(boundary_map_dict.items()):
                for j, n_path in enumerate(considered_operated_path_name):
                    if n_path in operated_n_path_dict:
                        unlimited_boundary_mat[i_path, j] = operated_n_path_dict[n_path]

            # collect informations
            boundary_map_matrix[n] = unlimited_boundary_mat
            boundary_mat_matrix_rank[n] = np.linalg.matrix_rank(unlimited_boundary_mat)
            allowed_path_idx_argument[n] = [1 if tpn in allowed_path[n-1] else 0 for tpn in considered_operated_path_name]

        return boundary_map_matrix, boundary_mat_matrix_rank, allowed_path_idx_argument

    def path_homology_for_connected_digraph(self, allowed_path, max_path):
        """_summary_ Calculate the dimension of the path homology group for required dimensions.

            Args:
                allowed_path (dict): the dict of all (0-n)-path, the format of the path is string.
                max_path (int): the maximum length of path, (maximum dimension for homology)

            Returns:
                betti_numbers (list): the Betti number for dimension 0 to dimension max_path
        """
        betti_numbers = np.array([0] * (max_path + 1))

        boundary_map_matrix, boundary_mat_matrix_rank, allowed_path_idx_argument =\
            self.utils_unlimited_boundary_operator(allowed_path, max_path)

        # betti_0 = dim(H(omega_0)) = dim(omega_0 or allowed_path_0) - rank(D0) - rank(D1)
        betti_0 = len(allowed_path[0]) - 0 - boundary_mat_matrix_rank[1]
        betti_numbers[0] = betti_0

        # When dim > 0, H_n = (A_n U ker(\partial)) / (A_n U \partial(A_n+1))
        for n in range(1, max_path+1):
            if len(allowed_path[n]) == 0:
                break

            # dim of (A_n U ker(\partial)) = dim A_n - rank(D_n)
            dim_0 = len(allowed_path[n]) - boundary_mat_matrix_rank[n]

            # dim of (A_n intersect \partial(A_n+1)) = dim A_n + rank(D_n+1) - dim(A_n U (D_n+1 B_n))
            # >>> Let W = (D_n+1 B_n); note: B_n here means the image space
            # >>> dim(A_n U W) = dim A_n + dim W - dim (A_n + W)

            # >>> dim (A_n + W) = rank([I_(A_n_argument); D_n+1])
            dim_An_Bn = np.linalg.matrix_rank(
                np.vstack(
                    [
                        np.eye(len(allowed_path_idx_argument[n+1])) * allowed_path_idx_argument[n+1],
                        boundary_map_matrix[n+1]
                    ]
                )
            )
            dim_1 = len(allowed_path[n]) + boundary_mat_matrix_rank[n+1] - dim_An_Bn

            betti_numbers[n] = dim_0 - dim_1

        return betti_numbers

    def path_homology(self, edges, nodes, max_path):
        # check the data type
        if edges.dtype != nodes.dtype:
            edges = edges.astype(str)
            nodes = nodes.astype(str)

        # split into independent components
        all_components = PathHomology.split_independent_compondent(edges, nodes)
        all_digraphs = PathHomology.split_independent_digraph(all_components, edges)

        betti_numbers = []
        for i_d, edges in enumerate(all_digraphs):
            if np.shape(edges)[1] <= 1:
                betti_numbers.append(np.array([1] + [0] * (max_path)))
            else:
                edges = PathHomology.remove_loops(edges)
                if np.shape(edges)[0] == 0:
                    betti_numbers.append(np.array([1] + [0] * (max_path)))
                    continue
                allowed_path = self.utils_generate_allowed_paths(edges, max_path)
                betti_numbers.append(self.path_homology_for_connected_digraph(allowed_path, max_path))
        return np.sum(betti_numbers, axis=0)

    def persistent_path_homology(self, cloudpoints, points_weight, max_path, filtration=None):
        """_summary_: Distance based filtration for cloudpoints.

            Modify:
                2022-04-25

            Args:
                cloudpoints (array): the coordinates of the points
                points_weight (list or array): the weights of point in the cloudpoints, used to define the digraph
                max_path (int): maximum path length, or maximum dimension of path homology
                filtration (array): distance-based filtration, default: array(0, max_distance, 0.1)

            Returns:
            all_betti_num (list): a list of betti numbers of the path homology groups in each
                                    dimension (0-max_path) obtained during the filtation.
        """
        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # initial
        distance_matrix = np.zeros([points_num, points_num], dtype=float)
        # fully connected map, [0: no edge, 1: out]
        fully_connected_map = np.zeros([points_num, points_num], dtype=int)
        for i in range(points_num):
            for j in range(points_num):
                if i == j:
                    continue
                distance = np.sqrt(np.sum((cloudpoints[i, :] - cloudpoints[j, :])**2))
                distance_matrix[i, j] = distance
                if points_weight[i] <= points_weight[j]:
                    fully_connected_map[i, j] = 1
        max_distance = np.max(distance_matrix)
        self.total_edges_num = np.sum(np.abs(fully_connected_map))
        self.max_distance = max_distance

        # filtration process
        if filtration is None:
            filtration = np.arange(0, np.round(max_distance, 2)+0.1, 0.1)

        all_betti_num = []
        save_time_flag = 0
        for n, snapshot_dis in enumerate(filtration):
            snapshot_map = np.ones([points_num]*2, dtype=int) * (distance_matrix <= snapshot_dis) * fully_connected_map

            start_ids = []
            end_ids = []
            for i in range(points_num):
                for j in range(points_num):
                    if i == j:
                        continue
                    if snapshot_map[i, j] == 1:
                        start_ids.append(i)
                        end_ids.append(j)
            edges = np.vstack([start_ids, end_ids]).T
            if save_time_flag == 1:
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            if np.shape(edges)[0] == self.total_edges_num:
                save_time_flag = 1
            betti_numbers = self.path_homology(edges, points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num

    def persistent_path_homology_from_digraph(
        self, cloudpoints, all_edges, max_path, filtration=None
    ):
        """_summary_: Distance-based filtration for digraph is performed.

            Modify:
                2022-04-25

            Args:
                cloudpoints (array): the coordinates of the points
                all_edges (array): the maximum edges of the final digraph, shape: [n, 2]
                filtration_angle_step (int, default: 30): the angle step during the angle-based filtration

            Returns:
            all_betti_num (list): a list of betti numbers of the path homology groups in each
                                    dimension (0-max_path) obtained during the angle-based filtation.
        """
        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # initial
        distance_matrix = np.zeros([points_num, points_num], dtype=float)
        for i in range(points_num-1):
            for j in range(i+1, points_num):
                distance = np.sqrt(np.sum((cloudpoints[i, :] - cloudpoints[j, :])**2))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        max_distance = np.max(distance_matrix)
        self.max_distance = max_distance

        # fully connected map, [0: no edge, 1: out]
        fully_connected_map = np.zeros([points_num, points_num], dtype=int)
        for i, one_edge in enumerate(all_edges):
            fully_connected_map[one_edge[0], one_edge[1]] = 1
        self.total_edges_num = np.sum(np.abs(fully_connected_map))

        # filtration process
        if filtration is None:
            filtration = np.arange(0, 10+0.1, 0.1)

        all_betti_num = []
        save_time_flag = 0
        for n, snapshot_dis in enumerate(filtration):
            snapshot_map = np.ones([points_num]*2, dtype=int) * (distance_matrix <= snapshot_dis) * fully_connected_map
            start_ids = []
            end_ids = []
            for i in range(points_num):
                for j in range(points_num):
                    if i == j:
                        continue
                    if snapshot_map[i, j] == 1:
                        start_ids.append(i)
                        end_ids.append(j)
            edges = np.vstack([start_ids, end_ids]).T
            if save_time_flag == 1:
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            if np.shape(edges)[0] == self.total_edges_num:
                save_time_flag = 1
            betti_numbers = self.path_homology(edges, points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num

    def persistent_angle_path_homology(self, cloudpoints, points_weight, max_path, filtration_angle_step=30):
        """_summary_: Angle-based filtration of cloudpoints is performed.
                      Specifically, we divide the 3-dimensional space in two steps. First, along the z-axis, for
                      any plane perpendicular to the xy-plane and passing through the z-axis will be divided into
                      12 sectors by rays passing through the origin at 30-degree intervals. Second, any of the
                      partitioned planes in the first step (usually the plane passing through the x-axis) will be
                      rotated along the z-axis at 30 degree intervals to de-partition the space and obtain 12 radial
                      regions perpendicular to the z-axis direction. Combining these two steps, the 3D space can be
                      divided into 72 (72 = 360/30 * 360/30 /2) regions. The angle filtration process is introduced
                      by considering all edges contained in the region in order.

            Args:
                cloudpoints (array): the coordinates of the points
                points_weight (list or array): the weights of point in the cloudpoints, used to define the digraph
                max_path (int): maximum path length, or maximum dimension of path homology
                filtration_angle_step (int, default: 30): the angle step during the angle-based filtration

            Returns:
            all_betti_num (list): a list of betti numbers of the path homology groups in each
                                    dimension (0-max_path) obtained during the angle-based filtation.
        """

        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # based on the initial xyz, to generate the filtration, degree
        filtration = []
        # angle between edge vector and z axis
        for vz in range(filtration_angle_step, 180+1, filtration_angle_step):
            # angle between edge vector and y
            for vy in [0*180, 1*180]:
                # angle between cross(edge_vector, z) and x axis
                for cross_vz_y in range(filtration_angle_step, 180+1, filtration_angle_step):
                    filtration.append([vz, cross_vz_y + vy])

        # initial vector
        initial_vector_x = np.array([0., 0., 0.])
        max_distance = 0
        initial_vector_y_idx = [0, 0]
        all_edge_vectors = []
        all_edge_idx = []  # from [0] -> [1]

        for i in range(points_num):
            for j in range(points_num):
                if i == j:
                    continue
                distance = np.sqrt(np.sum((cloudpoints[i, :] - cloudpoints[j, :])**2))
                edge_vector = cloudpoints[j, :] - cloudpoints[i, :]

                if distance > max_distance:
                    max_distance = distance
                    # record the vector idx corresponding to the max distance
                    initial_vector_y_idx = [i, j]

                if points_weight[i] <= points_weight[j]:
                    initial_vector_x += edge_vector
                    all_edge_vectors.append(edge_vector)
                    all_edge_idx.append([i, j])
        all_edge_idx = np.array(all_edge_idx)
        self.total_edges_num = len(all_edge_vectors)

        initial_vector_y = cloudpoints[initial_vector_y_idx[0], :] - cloudpoints[initial_vector_y_idx[1], :]
        # make sure the angle between x and y are acute angle
        if PathHomology().vector_angle(initial_vector_x, initial_vector_y) >= 90:
            initial_vector_y = -initial_vector_y
        initial_vector_z = np.cross(initial_vector_x, initial_vector_y)
        initial_vector_y = np.cross(initial_vector_x, initial_vector_z)  # make sure for orthogonal
        if self.initial_axes is None:
            self.initial_vector_x = initial_vector_x
            self.initial_vector_y = initial_vector_y
            self.initial_vector_z = initial_vector_z

        # calculate the angle between edge vector and x, y, z and realted rules
        two_related_angles = []
        for e_i, edge_v in enumerate(all_edge_vectors):
            edge_z = PathHomology().vector_angle(edge_v, self.initial_vector_z)
            edge_cross_vz_x = PathHomology().vector_angle(
                np.cross(edge_v, self.initial_vector_z), self.initial_vector_x)
            edge_y_flag = PathHomology().vector_angle(edge_v, self.initial_vector_y) // 90

            two_related_angles.append([edge_z, edge_cross_vz_x + 180*edge_y_flag])
        two_related_angles = np.array(two_related_angles)

        # persistent betti num
        all_betti_num = []
        edges = np.zeros([0, 2])
        for n, snapshot_angle in enumerate(filtration):
            snapshot_map_idx = np.append(
                np.where(two_related_angles[:, 0] < snapshot_angle[0])[0],
                np.where(
                    (
                        (two_related_angles[:, 0] <= snapshot_angle[0]) * (
                            two_related_angles[:, 0] > snapshot_angle[0]-filtration_angle_step
                        ) * (two_related_angles[:, 1] <= snapshot_angle[1])
                    )
                )
            )
            edges_temp = all_edge_idx[snapshot_map_idx, :]

            # delete original set
            two_related_angles = np.delete(two_related_angles, snapshot_map_idx, axis=0)
            all_edge_idx = np.delete(all_edge_idx, snapshot_map_idx, axis=0)
            edges = np.vstack([edges, edges_temp])

            betti_numbers = self.path_homology(edges.astype(int), points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num

    def persistent_angle_path_homology_from_digraph(
        self, cloudpoints, all_edges, max_path, filtration_angle_step=30
    ):
        """_summary_: Angle-based filtration for digraph is performed.

            Args:
                cloudpoints (array): the coordinates of the points
                all_edges (array): the maximum edges of the final digraph, shape: [n, 2]
                filtration_angle_step (int, default: 30): the angle step during the angle-based filtration

            Returns:
            all_betti_num (list): a list of betti numbers of the path homology groups in each
                                    dimension (0-max_path) obtained during the angle-based filtation.
        """
        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # based on the initial xyz, to generate the filtration, degree
        filtration = []
        # angle between edge vector and z axis
        for vz in range(filtration_angle_step, 180+1, filtration_angle_step):
            # angle between edge vector and y
            for vy in [0*180, 1*180]:
                # angle between cross(edge_vector, z) and x axis
                for cross_vz_y in range(filtration_angle_step, 180+1, filtration_angle_step):
                    filtration.append([vz, cross_vz_y + vy])

        # initial vector
        initial_vector_x = np.array([0., 0., 0.])
        all_edge_vectors = []
        all_edge_idx = all_edges  # from [0] -> [1]
        for i_v, (s_v, e_v) in enumerate(all_edges):
            edge_vector = cloudpoints[e_v, :] - cloudpoints[s_v, :]
            initial_vector_x += edge_vector
            all_edge_vectors.append(edge_vector)
        self.total_edges_num = len(all_edge_vectors)

        # get new xyz axis
        max_distance = 0
        initial_vector_y_idx = [0, 0]
        # upper triangular matrix
        for i in range(points_num-1):
            for j in range(i+1, points_num):
                distance = np.sqrt(np.sum((cloudpoints[i, :] - cloudpoints[j, :])**2))
                if distance > max_distance:
                    max_distance = distance
                    # record the vector idx corresponding to the max distance
                    initial_vector_y_idx = [i, j]

        initial_vector_y = cloudpoints[initial_vector_y_idx[0], :] - cloudpoints[initial_vector_y_idx[1], :]
        # make sure the angle between x and y are acute angle
        if PathHomology().vector_angle(initial_vector_x, initial_vector_y) >= 90:
            initial_vector_y = -initial_vector_y
        initial_vector_z = np.cross(initial_vector_x, initial_vector_y)
        initial_vector_y = np.cross(initial_vector_x, initial_vector_z)  # make sure for orthogonal
        if self.initial_axes is None:
            self.initial_vector_x = initial_vector_x
            self.initial_vector_y = initial_vector_y
            self.initial_vector_z = initial_vector_z

        # calculate the angle between edge vector and x, y, z and realted rules
        two_related_angles = []
        for e_i, edge_v in enumerate(all_edge_vectors):
            edge_z = PathHomology().vector_angle(edge_v, self.initial_vector_z)
            edge_cross_vz_x = PathHomology().vector_angle(
                np.cross(edge_v, self.initial_vector_z), self.initial_vector_x)
            edge_y_flag = PathHomology().vector_angle(edge_v, self.initial_vector_y) // 90
            two_related_angles.append([edge_z, edge_cross_vz_x + 180*edge_y_flag])
        two_related_angles = np.array(two_related_angles)

        # persistent betti num
        all_betti_num = []
        edges = np.zeros([0, 2])
        for n, snapshot_angle in enumerate(filtration):
            snapshot_map_idx = np.append(
                np.where(two_related_angles[:, 0] < snapshot_angle[0])[0],
                np.where(
                    (
                        (two_related_angles[:, 0] <= snapshot_angle[0]) * (
                            two_related_angles[:, 0] > snapshot_angle[0]-filtration_angle_step
                        ) * (two_related_angles[:, 1] <= snapshot_angle[1])
                    )
                )
            )
            edges_temp = all_edge_idx[snapshot_map_idx, :]

            # delete original set
            two_related_angles = np.delete(two_related_angles, snapshot_map_idx, axis=0)
            all_edge_idx = np.delete(all_edge_idx, snapshot_map_idx, axis=0)
            edges = np.vstack([edges, edges_temp])
            betti_numbers = self.path_homology(edges.astype(int), points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num


def input_cloudpoints(data_file, save_path, args):
    import pandas as pd
    data = pd.read_csv(data_file, header=0, index_col=0).values
    cloudpoints = data[:, 0:-1]
    points_weight = data[:, -1]
    max_path = args.max_path
    if args.filtration_type == 'distance':
        PH = PathHomology()
        define_filtration = np.arange(0, args.max_distance, 0.1)
        betti_num_all = PH.persistent_path_homology(
            cloudpoints, points_weight, max_path, filtration=define_filtration)
        result = {
            'max_distance': PH.max_distance,
            'betti_num': betti_num_all,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    elif args.filtration_type == 'angle':
        PH = PathHomology()
        betti_num_all = PH.persistent_angle_path_homology(
            cloudpoints, points_weight, max_path)
        result = {
            'betti_num': betti_num_all,
            'initial_vector_x': PH.initial_x_vector,
            'initial_vector_y': PH.initial_y_vector,
            'initial_vector_z': PH.initial_z_vector,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    return None


def input_digraph(data_file, save_path, args):
    import pandas as pd
    data = pd.read_csv(data_file, header=0, index_col=0)
    col_name = list(data.columns)
    cloudpoints = data[col_name[0:-2]].dropna(axis=0).values
    start_n = data[col_name[-2]].dropna(axis=0).values
    end_n = data[col_name[-1]].dropna(axis=0).values
    all_edges = np.vstack([start_n, end_n]).T
    max_path = args.max_path
    if args.filtration_type == 'distance':
        PH = PathHomology()
        betti_num_all = PH.persistent_path_homology_from_digraph(
            cloudpoints, all_edges, max_path)
        result = {
            'max_distance': PH.max_distance,
            'betti_num': betti_num_all,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    elif args.filtration_type == 'angle':
        PH = PathHomology()
        betti_num_all = PH.persistent_angle_path_homology_from_digraph(
            cloudpoints, all_edges, max_path)
        result = {
            'betti_num': betti_num_all,
            'initial_vector_x': PH.initial_x_vector,
            'initial_vector_y': PH.initial_y_vector,
            'initial_vector_z': PH.initial_z_vector,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    return None


def main(args):
    if args.input_type == 'cloudpoints':
        input_cloudpoints(args.input_data, args.save_name, args)
    elif args.input_type == 'digraph':
        input_digraph(args.input_data, args.save_name, args)
    else:
        print('Interal program.')
    return None


def parse_args(args):
    parser = argparse.ArgumentParser(description='Angle, distance, -based persistent path homology')

    parser.add_argument('--input_type', default='No', type=str, choices=['cloudpoints', 'digraph', 'No'])
    parser.add_argument('--input_data', default='cloudpoints.csv', type=str,
                        help='If the input type is cloudpoints, the input data should be the csv file, which contians the '
                        'cloudpoints and weights with the shape n*m, the n means the number of the points, (m-1) is the '
                        'dimension of the points, the last column are treated as weights.'
                        'For the digraph, the format of the file is .csv. The contents of the file is cloudpoints and edges.'
                        'The last two columns are start point idx and end point idx of the edges. All indices are start from 0')
    parser.add_argument('--filtration_type', default='angle', type=str, choices=['angle', 'distance'])
    parser.add_argument('--max_distance', default=5, type=float, help='if filtration_type is angle, it will be ignored')
    parser.add_argument('--save_name', default='./', type=str)
    parser.add_argument('--max_path', default=2, type=int)
    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
