import uproot
import numpy as np
import torch

def read_root_file(root_file, tree_path, x_key, y_key, z_key, time_key, predicted_tracks_indexes):
    """
    Reads data from a ROOT file and returns a dictionary with the specified keys.
    """
    

    with uproot.open(root_file) as f:
        predicted_tracks_indexes = f[predicted_tracks_indexes].array(library='np')
        predicted_tracks_indexes = np.array([np.array(list(stl_vector)) for stl_vector in predicted_tracks_indexes[0]], dtype=object)


        tree = f[tree_path]
        data = {
            'x': tree[x_key].array(library='np'),
            'y': tree[y_key].array(library='np'),
            'z': tree[z_key].array(library='np'),
            'time': tree[time_key].array(library='np'),
            'predicted_tracks_indexes': predicted_tracks_indexes
        }
    return data

def filter_predicted_tracks(predicted_tracks_indexes, mask):
    """
    Filters predicted_tracks_indexes based on the mask.
    Removes indices from each track if the corresponding value in the mask is False.
    :param predicted_tracks_indexes: List of tracks (list of lists or array of arrays)
    :param mask: Boolean mask indicating valid indices
    :return: Filtered predicted_tracks_indexes
    """
    # Trova gli indici validi dalla maschera
    valid_indices = np.where(mask)[0]

    # Filtra ogni traccia in predicted_tracks_indexes
    filtered_tracks = []
    for track in predicted_tracks_indexes:
        # Mantieni solo gli indici validi
        filtered_track = [index for index in track if index in valid_indices]
        if filtered_track:  # Aggiungi solo tracce non vuote
            filtered_tracks.append(filtered_track)

    return np.array(filtered_tracks, dtype=object)

def build_input_for_develop (x, y, z, time,time_window,  predicted_tracks_indexes):
    """
    Builds input tensor for the develop visualization from the provided data.
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :param time: time values
    :param time_window: time window for filtering
    :param predicted_tracks_indexes: indexes of predicted tracks
    """

    x = np.concatenate(x)
    y =  np.concatenate(y)
    z =  np.concatenate(z)
    time =  np.concatenate(time)
    features = np.stack((x, y, z, time), axis=-1)
    features_tensor = torch.tensor(features, dtype=torch.float32)


    if time_window:
        start, end = time_window
        mask = (features_tensor[:, 3] >= start) & (features_tensor[:, 3] <= end)
        features_tensor_in_time_window = features_tensor[mask]

    predicted_tracks_indexes = filter_predicted_tracks(predicted_tracks_indexes, mask.numpy())
    features_tensor_in_time_window = torch.tensor(features_tensor_in_time_window, dtype=torch.float32)
    
    return features_tensor_in_time_window, predicted_tracks_indexes, features_tensor