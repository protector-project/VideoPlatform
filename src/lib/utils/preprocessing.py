import numpy as np
import pandas as pd


def load_MT(trajectories, scene):
    cols = ["frame", "trackId", "bb_left", "bb_top", "bb_width", "bb_height"]
    cols2drop = ["bb_left", "bb_top", "bb_width", "bb_height"]
    scene_df = pd.DataFrame(trajectories, columns=cols)
    data = []
    scene_df = scene_df.sort_values(by=["frame", "trackId"])
    # Calculate center point of bounding box
    scene_df["x"] = scene_df["bb_left"] + scene_df["bb_width"] / 2
    scene_df["y"] = scene_df["bb_top"] + scene_df["bb_height"] / 2
    scene_df = scene_df.drop(columns=cols2drop)
    scene_df["sceneId"] = scene
    # new unique id by combining scene_id and track_id
    scene_df["rec&trackId"] = [
        rec_id + "_" + str(track_id).zfill(4)
        for rec_id, track_id in zip(scene_df.sceneId, scene_df.trackId)
    ]
    data.append(scene_df)
    data = pd.concat(data, ignore_index=True)
    rec_trackId2metaId = {}
    for i, j in enumerate(data["rec&trackId"].unique()):
        rec_trackId2metaId[j] = i
    data["metaId"] = [rec_trackId2metaId[i] for i in data["rec&trackId"]]
    data = data.drop(columns=["rec&trackId"])

    return data


def mask_step(x, step):
    """
    Create a mask to only contain the step-th element starting from the first element. Used to downsample
    """
    mask = np.zeros_like(x)
    mask[::step] = 1
    return mask.astype(bool)


def downsample(df, step):
    """
    Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
    df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
    pedestrian (metaId)
    :param df: pandas DataFrame - necessary to have column 'metaId'
    :param step: int - step size, similar to slicing-step param as in array[start:end:step]
    :return: pd.df - downsampled
    """
    mask = df.groupby(["metaId"])["metaId"].transform(mask_step, step=step)
    return df[mask]


def filter_short_trajectories(df, threshold):
    """
    Filter trajectories that are shorter in timesteps than the threshold
    :param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
    :param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
    :return: pd.df with trajectory length over threshold
    """
    len_per_id = df.groupby(
        by="metaId", as_index=False
    ).count()  # sequence-length for each unique pedestrian
    idx_over_thres = len_per_id[
        len_per_id["frame"] >= threshold
    ]  # rows which are above threshold
    idx_over_thres = idx_over_thres[
        "metaId"
    ].unique()  # only get metaIdx with sequence-length longer than threshold
    df = df[
        df["metaId"].isin(idx_over_thres)
    ]  # filter df to only contain long trajectories
    return df


def groupby_sliding_window(x, window_size, stride):
    x_len = len(x)
    n_chunk = (x_len - window_size) // stride + 1
    idx = []
    metaId = []
    for i in range(n_chunk):
        idx += list(range(i * stride, i * stride + window_size))
        metaId += ["{}_{}".format(x.metaId.unique()[0], i)] * window_size
    # temp = x.iloc()[(i * stride):(i * stride + window_size)]
    # temp['new_metaId'] = '{}_{}'.format(x.metaId.unique()[0], i)
    # df = df.append(temp, ignore_index=True)
    df = x.iloc()[idx]
    df["newMetaId"] = metaId
    return df


def sliding_window(df, window_size, stride):
    """
    Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
    chunked trajectories are overlapping
    :param df: df
    :param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
    :param stride: timesteps to move from one trajectory to the next one
    :return: df with chunked trajectories
    """
    gb = df.groupby(["metaId"], as_index=False)
    df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
    df["metaId"] = pd.factorize(df["newMetaId"], sort=False)[0]
    df = df.drop(columns="newMetaId")
    df = df.reset_index(drop=True)
    return df


def split_at_fragment_lambda(x, frag_idx, gb_frag):
    """Used only for split_fragmented()"""
    metaId = x.metaId.iloc()[0]
    counter = 0
    if metaId in frag_idx:
        split_idx = gb_frag.groups[metaId]
        for split_id in split_idx:
            x.loc[split_id:, "newMetaId"] = "{}_{}".format(metaId, counter)
            counter += 1
    return x


def split_fragmented(df):
    """
    Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
    Formally, this is done by changing the metaId at the fragmented frame and below
    :param df: DataFrame containing trajectories
    :return: df: DataFrame containing trajectories without fragments
    """

    gb = df.groupby("metaId", as_index=False)
    # calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
    df["frame_diff"] = gb["frame"].diff().fillna(value=1.0).to_numpy()
    fragmented = df[
        df["frame_diff"] != 1.0
    ]  # df containing all the first frames of fragmentation
    gb_frag = fragmented.groupby("metaId")  # helper for gb.apply
    frag_idx = fragmented.metaId.unique()  # helper for gb.apply
    df["newMetaId"] = df["metaId"]  # temporary new metaId

    df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
    df["metaId"] = pd.factorize(df["newMetaId"], sort=False)[0]
    df = df.drop(columns="newMetaId")
    return df


def load_and_window_MT(
    step, window_size, stride, trajectories, scene, pickle_path=None
):
    """
    Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
    - Split fragmented trajectories
    - Downsample fps
    - Filter short trajectories below threshold=window_size
    - Sliding window with window_size and stride
    :param step (int): downsample factor, step=12.5 means 1fps and step=5 means 2.5fps on MT
    :param window_size (int): Timesteps for one window
    :param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
    :param pickle_path (str): Alternative to path+mode, if there already is a pickled version of the raw MT as df
    :return pd.df: DataFrame containing the preprocessed data
    """
    if pickle_path is not None:
        df = pd.read_pickle(pickle_path)
    else:
        df = load_MT(trajectories, scene)
    df = split_fragmented(df)  # split track if frame is not continuous
    df = downsample(df, step=step)
    df = filter_short_trajectories(df, threshold=window_size)
    df = sliding_window(df, window_size=window_size, stride=stride)

    return df
