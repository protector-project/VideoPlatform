import math
from typing import Tuple
import os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import itertools
import cv2


CLASSES = ['bicycle', 'bus', 'car', 'motor', 'person', 'truck', 'van']
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']
DEBUG_SAVE_IMAGE = None
ROOT_DIR = "/usr/src/app/"


def process_tracking_output(out_dir: str, filter_class=None, min_traject_len=50):
    list_trajectories = []
    list_class_id = []
    map_traject2file = {}
    for file in os.listdir(out_dir):
        trajectories = {}
        with open(os.path.join(out_dir, file), 'r') as f:
            for line in f:
                infos = line.strip().split()
                frame_id, track_id, bbox_left, bbox_top, bbox_w, bbox_h, class_id, conf = int(infos[0]), int(
                    infos[1]), int(infos[2]), int(infos[3]), int(infos[4]), int(infos[5]), int(infos[6]), float(
                    infos[7])
                if filter_class is not None and class_id != filter_class:
                    continue
                center_x, center_y = int((bbox_left + bbox_w) / 2), int((bbox_top + bbox_h) / 2)
                if track_id in trajectories:
                    trajectories[track_id]['values'].append([center_x, center_y, class_id])
                    if class_id in trajectories[track_id]['nb_classes']:
                        trajectories[track_id]['nb_classes'][class_id] += 1
                    else:
                        trajectories[track_id]['nb_classes'][class_id] = 1
                else:
                    trajectories[track_id] = {
                        'values': [[center_x, center_y, class_id]],
                        'nb_classes': {class_id: 1}}
        for track_id, tr in trajectories.items():
            major_class_id = -1
            max_nb_class = 0
            for class_id in tr['nb_classes']:
                if tr['nb_classes'][class_id] > max_nb_class:
                    max_nb_class = tr['nb_classes'][class_id]
                    major_class_id = class_id
            new_traject = [t for t in tr['values'] if t[2] == major_class_id]
            if len(new_traject) >= min_traject_len:
                map_traject2file[len(list_trajectories)] = {'track_id': track_id, 'file': file}
                list_trajectories.append(new_traject)
                list_class_id.append(major_class_id)

    return list_trajectories, list_class_id, map_traject2file


def cost_func(s: list, t: list) -> float:
    return math.sqrt(math.pow(s[0] - t[0], 2) + math.pow(s[1] - t[1], 2))
    # return abs(s[0] - t[0]) + abs(s[1] - t[1])


def pairwise_distance(x: list) -> np.ndarray:
    pool = mp.Pool(mp.cpu_count())
    dist_matrix = np.zeros((len(x), len(x)))
    print(f"len of trajectories = {len(x)} nb_cpu = {mp.cpu_count()}")
    args = [(x[i_a], x[i_b]) for i_a, i_b in list(itertools.combinations(range(len(x)), 2))]
    # results = pool.starmap(dtw_distance, args)
    results = pool.starmap(euclid_distance, args)
    for i_tuple, re in zip([(i_a, i_b) for i_a, i_b in list(itertools.combinations(range(len(x)), 2))], results):
        # unpack
        i_a, i_b = i_tuple
        # set it in the matrix
        dist_matrix[i_b][i_a] = re
        dist_matrix[i_a][i_b] = re

    return dist_matrix


def euclid_distance(s: list, t: list):
    cost = 0
    for i in range(min(len(s), len(t))):
        cost += cost_func(s[i], t[i])
    return cost


def dtw_distance(s: list, t: list, w: int = 3) -> float:
    # print(f"Calculating dtw_distance between {i_s} and {i_t}")
    m, l = len(s), len(t)
    # w = np.max([window, abs(m - l)])
    dtw_matrix = np.zeros((m + 1, l + 1))
    for k in range(m + 1):
        dtw_matrix[k, 0] = np.inf
    for k in range(l + 1):
        dtw_matrix[0, k] = np.inf

    dtw_matrix[0, 0] = 0

    for k in range(1, m + 1):
        for p in range(np.max([k - w, 1]), np.min([k + w, l]) + 1):
            cost = cost_func(s[k], t[p])
            dtw_matrix[k, p] = cost + min(dtw_matrix[k - 1, p], dtw_matrix[k, p - 1], dtw_matrix[k - 1, p - 1])

    return dtw_matrix[m, l]


def vat(data: list, return_odm: bool = False, figure_size: Tuple = (10, 10), save_fig=None):
    """VAT means Visual assessment of tendency. basically, it allow to asses cluster tendency
    through a map based on the dissimilarity matrix.
    Parameters
    ----------
    data : matrix
        numpy array
    return_odm : return the Ordered dissimilarity Matrix
        boolean (default to False)
    figure_size : size of the VAT.
        tuple (default to (10,10))
    save_fig: name of figure
        string
    Return
    -------
    ODM : matrix
        the ordered dissimilarity matrix plotted.
    """

    ordered_dissimilarity_matrix, _ = compute_ordered_dissimilarity_matrix(data)

    _, ax = plt.subplots(figsize=figure_size)
    ax.imshow(
        ordered_dissimilarity_matrix,
        cmap="gray",
        vmin=0,
        vmax=np.max(ordered_dissimilarity_matrix),
    )
    if save_fig is not None:
        plt.savefig(save_fig)
    else:
        plt.show()

    if return_odm is True:
        return ordered_dissimilarity_matrix


def compute_ordered_dis_njit(matrix_of_pairwise_distance: np.ndarray):  # pragma: no cover
    """
    The ordered dissimilarity matrix is used by visual assessment of tendency. It is a just a a reordering
    of the dissimilarity matrix.
    Parameter
    ----------
    x : matrix
        numpy array
    Return
    -------
    ODM : matrix
        the ordered dissimilarity matrix
    """

    # Step 1 :

    observation_path = np.zeros(matrix_of_pairwise_distance.shape[0], dtype="int")
    list_of_int = np.zeros(matrix_of_pairwise_distance.shape[0], dtype="int")
    # nearest_point = np.zeros(matrix_of_pairwise_distance.shape[0], dtype="int")
    edge_values = np.zeros(matrix_of_pairwise_distance.shape[0], dtype="float")

    index_of_maximum_value = np.argmax(matrix_of_pairwise_distance)

    column_index_of_maximum_value = (
            index_of_maximum_value // matrix_of_pairwise_distance.shape[1]
    )

    list_of_int[0] = column_index_of_maximum_value
    observation_path[0] = column_index_of_maximum_value
    # nearest_point[0] = column_index_of_maximum_value
    edge_values[0] = 0

    K = np.linspace(
        0,
        matrix_of_pairwise_distance.shape[0] - 1,
        matrix_of_pairwise_distance.shape[0],
    ).astype(np.int32)

    J = np.delete(K, column_index_of_maximum_value)

    for r in range(1, matrix_of_pairwise_distance.shape[0]):

        p, q = (-1, -1)

        mini = np.max(matrix_of_pairwise_distance)

        for candidate_p in observation_path[0:r]:
            for candidate_j in J:
                if matrix_of_pairwise_distance[candidate_p, candidate_j] < mini:
                    p = candidate_p
                    q = candidate_j
                    mini = matrix_of_pairwise_distance[p, q]

        list_of_int[r] = q
        observation_path[r] = q
        # nearest_point[r] = p
        edge_values[r] = mini
        ind_q = np.where(J == q)[0][0]
        J = np.delete(J, ind_q)

    # Step 3

    ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)

    for column_index_of_maximum_value in range(ordered_matrix.shape[0]):
        for j in range(ordered_matrix.shape[1]):
            ordered_matrix[
                column_index_of_maximum_value, j
            ] = matrix_of_pairwise_distance[
                list_of_int[column_index_of_maximum_value], list_of_int[j]
            ]

    # Step 4 :
    # , nearest_point, edge_values
    return ordered_matrix, list_of_int, edge_values


def plot_trajectories(x: list, ax, color=None):
    for i, traject in enumerate(x):
        x_axis = [pnt[0] for pnt in traject]
        y_axis = [pnt[1] for pnt in traject]
        if color is not None:
            ax.plot(x_axis, y_axis, color=color)
        else:
            ax.plot(x_axis, y_axis, color=COLORS[traject[0][2]])


def compute_cluster_threshold(edge_values: list):
    sorted_values = sorted(edge_values)
    # if DEBUG_SAVE_IMAGE is not None:
    #     x_axis = [edge_values[i] for i in range(1, len(edge_values))]
    #     plt.plot(range(len(edge_values) - 1), x_axis, '-bo')
    #     plt.savefig(f"{DEBUG_SAVE_IMAGE}edge_values.png")
    #
    #     x_axis = [sorted_values[i] for i in range(1, len(sorted_values))]
    #     plt.plot(range(len(sorted_values) - 1), x_axis, '-bo')
    #     plt.savefig(f"{DEBUG_SAVE_IMAGE}edge_values_sorted.png")

    max_value = -1
    max_index = -1
    for i in range(len(sorted_values)):
        p2p1 = [len(sorted_values) - 1, sorted_values[len(sorted_values) - 1] - sorted_values[0]]
        p1p3 = [-i, sorted_values[0] - sorted_values[i]]
        dis = np.linalg.norm(np.cross(p2p1, p1p3)) / np.linalg.norm(p2p1)
        if dis > max_value:
            max_value = dis
            max_index = i
    print(f"elbow index = {max_index} value = {sorted_values[max_index]}")
    return (sorted_values[max_index] + sorted_values[max_index + 1]) / 2


def compute_ordered_dissimilarity_matrix(x: list):
    print("calculating dtw pairwise distance ....")
    matrix_of_pairwise_distance = pairwise_distance(x)
    print("computing ordered dissimilarity matrix ....")
    dis_matrix, spanning_tree, edge_values = compute_ordered_dis_njit(matrix_of_pairwise_distance)
    alpha = compute_cluster_threshold(edge_values)

    lst_cluster = []
    cluster = [spanning_tree[0]]
    for i in range(1, len(spanning_tree)):
        if edge_values[i] > alpha:
            lst_cluster.append(cluster)
            cluster = [spanning_tree[i]]
        else:
            cluster.append(spanning_tree[i])
    lst_cluster.append(cluster)
    for i, cl in enumerate(lst_cluster):
        print(f"cluster {i}: {cl}")
    return dis_matrix, lst_cluster


def ivat(data: list, return_odm: bool = False, figure_size: Tuple = (10, 10), save_fig=None):
    """iVat return a visualisation based on the Vat but more reliable and easier to
    interpret.
    Parameters
    ----------
    data : matrix
        numpy array
    return_odm : return the Ordered dissimilarity Matrix
            boolean (default to False)
    figure_size : size of the VAT.
        tuple (default to (10,10))
    save_fig: name of figure
        string
    Return
    -------
    D_prim : matrix
        the ivat ordered dissimilarity matrix
    """

    ordered_matrix = compute_ivat_ordered_dissimilarity_matrix(data)

    _, ax = plt.subplots(figsize=figure_size)
    ax.imshow(ordered_matrix, cmap="gray", vmin=0, vmax=np.max(ordered_matrix))
    if save_fig is not None:
        plt.savefig("ivat.png")
    else:
        plt.show()

    if return_odm is True:
        return ordered_matrix


def compute_ivat_ordered_dissimilarity_matrix(x: list):
    """The ordered dissimilarity matrix is used by ivat. It is a just a a reordering
    of the dissimilarity matrix.
    Parameters
    ----------
    x : matrix
        numpy array
    Return
    -------
    D_prim : matrix
        the ordered dissimilarity matrix
    """

    ordered_matrix, _ = compute_ordered_dissimilarity_matrix(x)
    print("Re-ordering matrix ...")
    re_ordered_matrix = np.zeros((ordered_matrix.shape[0], ordered_matrix.shape[0]))

    for r in range(1, ordered_matrix.shape[0]):
        # Step 1 : find j for which D[r,j] is minimum and j ipn [1:r-1]

        j = np.argmin(ordered_matrix[r, 0:r])

        # Step 2 :

        re_ordered_matrix[r, j] = ordered_matrix[r, j]
        re_ordered_matrix[j, r] = ordered_matrix[r, j]

        # Step 3 : for c : 1, r-1 with c !=j
        c_tab = np.array(range(0, r))
        c_tab = c_tab[c_tab != j]

        for c in c_tab:
            re_ordered_matrix[r, c] = max(ordered_matrix[r, j], re_ordered_matrix[j, c])
            re_ordered_matrix[c, r] = re_ordered_matrix[r, c]

    return re_ordered_matrix


def write_result_to_video(result):
    for file_name in result:
        print(f"Write result for {file_name} ....")
        tracking_info = {}
        with open(os.path.join(ROOT_DIR, f"out/pza/trajectories/{file_name}.txt"), 'r') as f:
            for line in f:
                infos = line.strip().split()
                frame_id, track_id, bbox_left, bbox_top, bbox_w, bbox_h, class_id, conf = int(infos[0]), int(
                    infos[1]), int(infos[2]), int(infos[3]), int(infos[4]), int(infos[5]), int(infos[6]), float(
                    infos[7])
                if frame_id in tracking_info:
                    tracking_info[frame_id].append([track_id, bbox_left, bbox_top, bbox_w, bbox_h])
                else:
                    tracking_info[frame_id] = [[track_id, bbox_left, bbox_top, bbox_w, bbox_h]]
        cap = cv2.VideoCapture(os.path.join(ROOT_DIR, f"out/pza/{file_name}_original.mp4"))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(ROOT_DIR, f"out/anomaly/{file_name}_anomaly_detection.mp4"), fourcc, fps, (w, h))
        i_frame = 0
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                if i_frame in tracking_info:
                    for track in tracking_info[i_frame]:
                        if track[0] in result[file_name]:
                            intbox = tuple(map(int, (track[1], track[2], track[1]+track[3], track[2]+track[4])))
                            if "ANOMALY" in result[file_name][track[0]]:
                                frame = cv2.rectangle(
                                    frame, intbox[0:2], intbox[2:4], color=(0, 0, 255), thickness=3)
                                frame = cv2.putText(frame, result[file_name][track[0]], (intbox[0], intbox[1] + 30),
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)
                            else:
                                frame = cv2.rectangle(
                                    frame, intbox[0:2], intbox[2:4], color=(0, 255, 0), thickness=3)
                                frame = cv2.putText(frame, result[file_name][track[0]], (intbox[0], intbox[1] + 30),
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), thickness=2)
                writer.write(frame)
                i_frame += 1
            else:
                break
        cap.release()
        writer.release()


def main():
    beta = 0.01
    cluster_result = {}
    for i_c in range(len(CLASSES)):
        list_traject, lst_class_id, map_traject = process_tracking_output(ROOT_DIR + "out/pza/trajectories", filter_class=i_c)
        if len(list_traject) == 0:
            continue
        nb_min_cluster_amount = beta * len(list_traject)
        print(f"nb_min_cluster_amount = {nb_min_cluster_amount}")
        _, ax = plt.subplots()
        plot_trajectories(list_traject, ax)
        plt.savefig(f"{ROOT_DIR}/out/trajectories_{CLASSES[i_c]}.png")
        _, clusters = compute_ordered_dissimilarity_matrix(list_traject)

        _, ax = plt.subplots()
        for i, cl in enumerate(clusters):
            traject = []
            for c in cl:
                traject.append(list_traject[c])
                file_name = map_traject[c]['file'].replace(".txt", "")
                if file_name not in cluster_result:
                    cluster_result[file_name] = {}
                if len(cl) < nb_min_cluster_amount:
                    print(f"{'!' * 10} DETECTED ANOMALY IN {file_name} {'!' * 10}")
                    cluster_result[file_name][map_traject[c]['track_id']] = f"{CLASSES[i_c]}_ANOMALY"
                else:
                    cluster_result[file_name][map_traject[c]['track_id']] = f"{CLASSES[i_c]}_cluster_{i}"

            plot_trajectories(traject, ax,
                              color=COLORS[int(i % len(COLORS))] if len(cl) > nb_min_cluster_amount else 'black')
        plt.savefig(f"{ROOT_DIR}/out/cluster_{CLASSES[i_c]}.png")
        print(f"{'-'*10} DONE CLASS {CLASSES[i_c]} {'-'*10}")
    write_result_to_video(cluster_result)


if __name__ == '__main__':
    main()
