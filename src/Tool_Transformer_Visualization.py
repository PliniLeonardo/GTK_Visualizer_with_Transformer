
from itertools import combinations_with_replacement
import numpy as np
import torch
from src.Transformer import *
import plotly.graph_objects as go
import plotly.io as pio
import uproot
import matplotlib.pyplot as plt
import yaml
import os
import pandas as pd


def read_root_file(file_path, tree_path, x_path, y_path, z_path, time_path):
    '''
    Docstring per read_root_file
    
    :param file_path: path to the root file
    :param tree_path: path to the tree inside the root file
    :param keys: list of keys to extract
    :return: dictionary with keys and their corresponding numpy arrays
    '''
    tree = uproot.open(file_path)[tree_path]
    x = tree[x_path].array(library="np")
    y = tree[y_path].array(library="np")
    z = tree[z_path].array(library="np")
    time = tree[time_path].array(library="np")
    data = {
        'x': x,
        'y': y,
        'z': z,
        'time': time
    }
    return data


def build_input_for_transformer(x, y, z, time, time_window):
    '''
    Docstring per build_input_for_transformer
    
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :param time: time values
    :param time_window: time window for filtering
    '''

    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    time = np.concatenate(time)
    features = np.stack((x, y, z, time), axis=-1)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Apply time window filter
    if time_window:
        start, end = time_window
        mask = (features_tensor[:, 3] >= start) & (features_tensor[:, 3] <= end)
        features_tensor = features_tensor[mask]

    z_mapping = {79575: 0, 79625: 1, 86820: 2, 102400: 3}
    z_column = features_tensor[:, 2]
    mapped_z = torch.tensor([z_mapping.get(z.item(), -1) for z in z_column])
    features_tensor[:, 2] = mapped_z

    # order the input using the gtk station to fit the transformer's expectations
    features_tensor = features_tensor[features_tensor[:, 2].argsort()]
    num_of_hits = features_tensor.shape[0]
    pair_indices = list(combinations_with_replacement(range(num_of_hits), 2))
    pair_indices = torch.tensor(pair_indices, dtype=torch.long)

    return features_tensor, pair_indices


def transformer_model_load():
    yaml_path = "src/Transformer.yaml"
    checkpoint_path = "src/Transformer_GTKTime-KTAG_V4.ckpt"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    Lightning_model = LightningEncoder(config)
    ckpt = torch.load(checkpoint_path,map_location=torch.device('cpu') )
    state_dict = ckpt["state_dict"]      
    Lightning_model.load_state_dict(state_dict, strict=True)

    torch_model = EncoderONNX(config)
    lightning_state_dict = Lightning_model.state_dict()
    torch_state_dict = {}
    for key, value in lightning_state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
            torch_state_dict[new_key] = value

    torch_model.load_state_dict(torch_state_dict, strict=True)
    torch_model.eval()
    print("Loaded Transformer model from checkpoint")
    return torch_model


def edge_dictionary_transformer(model_output, threshold, pair_indices):
    """
    Creates a dictionary of edges with scores above a given threshold.

    Args:
        model_output (torch.Tensor): Output of the model.
        threshold (float): Minimum score to include an edge.
        pair_indices (torch.Tensor): Tensor of pair indices.

    Returns:
        dict: Dictionary {(i, j): score} with edges and their scores.
    """
    # Extract edges
    src = [edge[0] for edge in pair_indices]
    dst = [edge[1] for edge in pair_indices]
    scores = model_output[:, :, 1].squeeze()  # Assuming the second column is the score

    edge_dict = {}
    for i in range(len(src)):
        if scores[i] > threshold:
            # Detach the score to remove gradients
            edge_dict[(src[i], dst[i])] = scores[i].detach()

    return edge_dict

def allowed_edges(input_tensor, edge_dict):
    """
    Returns only the connections between nodes belonging to allowed GTK stations.

    Args:
        input_tensor (torch.Tensor): Tensor where the third column represents GTK stations (0, 1, 2, 3).
        edge_dict (dict): Dictionary {(i, j): score} representing connections between nodes.

    Returns:
        dict: Filtered edge_dict containing only connections between allowed GTK stations.
    """
    # Map node → station (using the third column of the tensor)
    node_to_station = {int(node): int(input_tensor[node, 2].item()) for node in range(input_tensor.shape[0])}

    # Allowed station pairs
    allowed_station_pairs = {(0, 1), (1, 2), (0, 2), (2, 3)}

    # Filter connections to include only those between allowed station pairs
    return {
        (int(i.item()), int(j.item())): score
        for (i, j), score in edge_dict.items()
        if (node_to_station.get(int(i.item())), node_to_station.get(int(j.item()))) in allowed_station_pairs
    }


def best_exiting(edge_dict):
    """
    Seleziona il miglior edge uscente per ciascun nodo sorgente, in base allo score.

    Args:
        edge_dict (dict): Dizionario {(i, j): score} con connessioni tra nodi e punteggi.

    Returns:
        dict: Dizionario {(i, j): score} contenente solo l'edge uscente con score più alto per ogni nodo sorgente i.
    """
    best_exiting_edges = {}
    for (i, j), score in edge_dict.items():
        if not any(k[0] == i and v >= score for k, v in best_exiting_edges.items()):
            # Rimuove eventuale edge precedente con stesso nodo sorgente
            best_exiting_edges = {k: v for k, v in best_exiting_edges.items() if k[0] != i}
            best_exiting_edges[(i, j)] = score
    return best_exiting_edges

def best_entering_edges_with_gtk_correction(best_exiting_edges, features_tensor):
    """
    Applies GTK correction to the best outgoing edges, handling ambiguous cases between GTK stations.

    Args:
        best_exiting_edges (dict): {(i, j): score} with the best outgoing edges.
        features_tensor (torch.Tensor): Tensor where the third column represents GTK stations (0, 1, 2, 3).

    Returns:
        dict: Corrected best entering edges according to GTK logic.
    """
    # Group nodes by their GTK station (from the third column of features_tensor)
    gtk_stations = {station: (features_tensor[:, 2] == station).nonzero(as_tuple=True)[0].tolist()
                    for station in range(4)}

    best_entering_edges = {}

    # Iterate over destination nodes
    for e_node in set(k[1] for k in best_exiting_edges.keys()):
        # Filter edges that enter the current destination node
        filtered_edges = {k: v for k, v in best_exiting_edges.items() if k[1] == e_node}
        sorted_edges = sorted(filtered_edges.items(), key=lambda x: x[1], reverse=True)  # Sort by score descending

        if len(sorted_edges) >= 2:
            # Top two edges with the highest scores
            (edge1, score1), (edge2, score2) = sorted_edges[:2]
            src1, src2 = edge1[0], edge2[0]

            # Case 1: 0→2 and 1→2
            if e_node in gtk_stations[2] and src1 in gtk_stations[0] and src2 in gtk_stations[1]:
                best_entering_edges[(src1, src2)] = score1  # 0→1
                best_entering_edges[edge2] = score2         # 1→2

            # Case 2: 1→2 and 0→2
            elif e_node in gtk_stations[2] and src1 in gtk_stations[1] and src2 in gtk_stations[0]:
                best_entering_edges[(src2, src1)] = score2  # 0→1
                best_entering_edges[edge1] = score1         # 1→2

            # Other cases (e.g., on gtk3): take only the best edge
            else:
                best_entering_edges[edge1] = score1

        elif len(sorted_edges) == 1:
            edge, score = sorted_edges[0]
            best_entering_edges[edge] = score

    return best_entering_edges

def find_sequences(pairs):
    '''
    this function create the sequence of a track starting from a dictionary of pairs
    '''
    
    # Create a dictionary to map the first element to the second element
    connection_map = {first: second for first, second in pairs}

    # Initialize an empty list to store the sequences
    sequences = []

    # To find starting points, find elements that are not second elements in any tuple
    all_second_elements = set(connection_map.values())
    starting_points = [first for first in connection_map if first not in all_second_elements]

    # Iterate through starting points to form sequences
    for start in starting_points:
        sequence = [start]
        while start in connection_map:
            start = connection_map[start]
            sequence.append(start)
        sequences.append(sequence)

    return sequences

def make_pred_tracks(edge_dict, features_tensor):
    """
    This function takes a batch of data, model logits, and a threshold to create predicted tracks.

    Args:
        edge_dict (dict): Dictionary {(i, j): score} representing connections between nodes.
        features_tensor (torch.Tensor): Tensor where the third column represents GTK stations (0, 1, 2, 3).

    Returns:
        list: List of predicted tracks (sequences of nodes).
    """
    # 3.1 Filter edges to include only allowed GTK station pairs
    edge_dict = allowed_edges(features_tensor, edge_dict)
    
    # 3.2 Select the best outgoing edge per source node
    best_exiting_edges = best_exiting(edge_dict)
    
    # 3.3 Apply GTK correction to the best outgoing edges
    best_entering_edges = best_entering_edges_with_gtk_correction(best_exiting_edges, features_tensor)
    
    # 3.4 Build tracks with the remaining edges
    pred_tracks = find_sequences(best_entering_edges.keys())

    return pred_tracks



def plot_gtk_hits_from_tensor(features_tensor, plot_folder_path):
    """
    Plots the hits on 4 different GTK planes using features_tensor, highlighting overlapping hits
    and positioning times around the hits to avoid overlap.

    Args:
        features_tensor (torch.Tensor): Tensor with columns:
            - x: x-coordinate of the hit
            - y: y-coordinate of the hit
            - gtk_station: GTK station (0, 1, 2, 3)
            - t: time of the hit

    Returns:
        None
    """
    # Convert tensor to numpy array for easier manipulation
    features = features_tensor.numpy()

    # Extract columns
    x = features[:, 0]
    y = features[:, 1]
    gtk_stations = features[:, 2].astype(int)
    t = features[:, 3]

    # Create subplots for the 4 GTK stations in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))  # Adjust figsize for larger plots
    axs = axs.flatten()  # Flatten the 2x2 grid for easier indexing

    # Definisci i limiti fissi per X e Y
    x_lim = (-30.4, 30.4)
    y_lim = (-13.5, 13.5)

    # Set unique markers for each node
    num_nodes = len(x)
    markers = ['o', 'v', '<', '>', '^', 's', 'P', 'X', 'H', 'd'] * 20
    node_markers = {i: markers[i % len(markers)] for i in range(num_nodes)}

    # Function to calculate radial distance
    def radial_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Loop through each GTK station and plot
    for gtk in range(4):
        gtk_hits = features[gtk_stations == gtk]
        for i, hit in enumerate(gtk_hits):
            # Check for nearby hits (distance ≤ 1)
            distances = np.sqrt((gtk_hits[:, 0] - hit[0])**2 + (gtk_hits[:, 1] - hit[1])**2)
            close_hits = distances <= 1

            if close_hits.sum() > 1:  # Highlight overlapping hits
                axs[gtk].scatter(hit[0], hit[1],
                                 c='red',  # Highlight color
                                 marker=node_markers[i],
                                 s=100, alpha=0.6, edgecolors='black')
            else:  # Normal hits
                axs[gtk].scatter(hit[0], hit[1],
                                 c='blue',  # Default color
                                 marker=node_markers[i],
                                 s=100, edgecolors='black')

            # Position times around the hit
            nearby_hits = gtk_hits[close_hits]
            offsets = [(0.4, 0.4), (-0.4, -0.4), (-0.4, 0.4), (0.4, -0.4)]  # Top-right, bottom-left, top-left, bottom-right
            for j, nearby_hit in enumerate(nearby_hits):
                if j < len(offsets):  # Limit to 4 positions
                    dx, dy = offsets[j]
                    axs[gtk].text(nearby_hit[0] + dx, nearby_hit[1] + dy, round(nearby_hit[3], 2), fontsize=10)

        axs[gtk].set_xlabel('x')
        axs[gtk].set_ylabel('y')
        axs[gtk].set_title(f'GTK{gtk}', pad=20)
        axs[gtk].grid()
        axs[gtk].set_xlim(x_lim)  # Imposta i limiti fissi per X
        axs[gtk].set_ylim(y_lim)  # Imposta i limiti fissi per Y

    plt.tight_layout()  # Adjust layout to prevent overlap
    # plt.show()

    # Save plot into the folder "plots"
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)
    fig.savefig(os.path.join(plot_folder_path, 'GTK_hits_visualization.png'), dpi=300)
    plt.close(fig)


def split_hits_by_gtk(features_tensor, dataframe_path):
    """
    Splits the hits in features_tensor into 4 DataFrames, one for each GTK station.

    Args:
        features_tensor (torch.Tensor): Tensor with columns:
            - x: x-coordinate of the hit
            - y: y-coordinate of the hit
            - z: z-coordinate of the hit (original GTK station mapping)
            - time: time of the hit

    Returns:
        dict: A dictionary with keys 'GTK0', 'GTK1', 'GTK2', 'GTK3' and values as DataFrames.
    """
    # Convert tensor to numpy array for easier manipulation
    features = features_tensor.numpy()

    # Extract columns
    x = features[:, 0]
    y = features[:, 1]
    z = features[:, 2]
    time = features[:, 3]

    # Create a DataFrame from the features
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'time': time})

    # Split the DataFrame by GTK station
    gtk_dfs = {}
    for gtk in range(4):
        gtk_dfs[f'GTK{gtk}'] = df[df['z'] == gtk].reset_index(drop=True)

    # Save DataFrames into the specified folder
    if not os.path.exists(dataframe_path):
        os.makedirs(dataframe_path)
    for gtk, gtk_df in gtk_dfs.items():
        gtk_df.to_csv(os.path.join(dataframe_path, f'{gtk}_hits.csv'), index=False)
    
    return gtk_dfs

 
# def plot_3d_interactive_transformer(pred_tracks, features_tensor, save_path,
#                         marker_size=4, line_width=3, show=True):
#     """
#     Interactive 3D Plotly plot of predicted tracks. Unassigned hits drawn as black points.
#     Args:
#         pred_tracks: list of tracks (each track = list/array of hit indices)
#         features_tensor: torch.Tensor or numpy array with columns [x, y, z_station, (t)]
#     """

#     # features -> numpy
#     features = features_tensor.numpy() if hasattr(features_tensor, "numpy") else np.asarray(features_tensor)
#     x = features[:, 0]
#     y = features[:, 1]
#     z_station = features[:, 2].astype(int)

#     fig = go.Figure()
#     base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'grey', 'pink']

#     # plot tracks
#     for i, track in enumerate(pred_tracks):
#         idx = np.asarray(track, dtype=int)
#         if idx.size == 0:
#             continue
#         fig.add_trace(go.Scatter3d(
#             x=x[idx], y=z_station[idx], z=y[idx],
#             mode='lines+markers',
#             line=dict(width=line_width, color=base_colors[i % len(base_colors)]),
#             marker=dict(size=marker_size),
#             name=f"track_{i}"
#         ))

#     # unassigned hits (black)
#     if len(pred_tracks) > 0:
#         used = np.unique(np.concatenate([np.asarray(t, dtype=int) for t in pred_tracks if len(t) > 0])).astype(int)
#     else:
#         used = np.array([], dtype=int)
#     mask_unused = np.ones(len(x), dtype=bool)
#     if used.size > 0:
#         mask_unused[used] = False
#     if mask_unused.sum() > 0:
#         fig.add_trace(go.Scatter3d(
#             x=x[mask_unused], y=z_station[mask_unused], z=y[mask_unused],
#             mode='markers',
#             marker=dict(size=max(2, marker_size - 1), color='black'),
#             name='unassigned'
#         ))

#     # station planes
#     x_lim = (-30.4, 30.4)  # Limiti fissi per X
#     y_lim = (-13.5, 13.5)  # Limiti fissi per Y
#     X_plane = [x_lim[0], x_lim[1], x_lim[1], x_lim[0]]
#     Z_plane = [y_lim[0], y_lim[0], y_lim[1], y_lim[1]]
#     for s in sorted(np.unique(z_station)):
#         fig.add_trace(go.Mesh3d(
#             x=X_plane, y=[s] * 4, z=Z_plane,
#             i=[0, 0], j=[1, 2], k=[2, 3],
#             opacity=0.12, color='black', showlegend=False
#         ))

#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(title='X', range=x_lim),  # Imposta i limiti fissi per X
#             yaxis=dict(title='GTK Station'),
#             zaxis=dict(title='Y', range=y_lim),  # Imposta i limiti fissi per Y
#             aspectmode='auto'
#         ),
#         margin=dict(l=0, r=0, t=40, b=0),
#         title="Interactive predicted tracks"
#     )

#     pio.write_html(fig, file=save_path, auto_open=False, include_plotlyjs='cdn')
#     if show:
#         fig.show()
#     return fig

def plot_3d_interactive_transformer(pred_tracks, features_tensor, save_path,
                                    marker_size=4, line_width=3, show=True):
    """
    Interactive 3D Plotly plot of predicted tracks. Unassigned hits drawn as black points.
    Args:
        pred_tracks: list of tracks (each track = list/array of hit indices)
        features_tensor: torch.Tensor or numpy array with columns [x, y, z_station, (t)]
    """

    # features -> numpy
    features = features_tensor.numpy() if hasattr(features_tensor, "numpy") else np.asarray(features_tensor)
    x = features[:, 0]
    y = features[:, 1]
    z_station = features[:, 2].astype(int)

    fig = go.Figure()
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'grey', 'pink']

    # plot tracks
    for i, track in enumerate(pred_tracks):
        idx = np.asarray(track, dtype=int)
        if idx.size == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=x[idx], y=z_station[idx], z=y[idx],
            mode='lines+markers',
            line=dict(width=line_width, color=base_colors[i % len(base_colors)]),
            marker=dict(size=marker_size),
            name=f"Track {i}",
            visible=True  # Default: all tracks visible
        ))

    # unassigned hits (black)
    if len(pred_tracks) > 0:
        used = np.unique(np.concatenate([np.asarray(t, dtype=int) for t in pred_tracks if len(t) > 0])).astype(int)
    else:
        used = np.array([], dtype=int)
    mask_unused = np.ones(len(x), dtype=bool)
    if used.size > 0:
        mask_unused[used] = False
    if mask_unused.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=x[mask_unused], y=z_station[mask_unused], z=y[mask_unused],
            mode='markers',
            marker=dict(size=max(2, marker_size - 1), color='black'),
            name='Unassigned Hits',
            visible=True  # Always visible
        ))

    # station planes
    x_lim = (-30.4, 30.4)  # Limiti fissi per X
    y_lim = (-13.5, 13.5)  # Limiti fissi per Y
    X_plane = [x_lim[0], x_lim[1], x_lim[1], x_lim[0]]
    Z_plane = [y_lim[0], y_lim[0], y_lim[1], y_lim[1]]
    for s in sorted(np.unique(z_station)):
        fig.add_trace(go.Mesh3d(
            x=X_plane, y=[s] * 4, z=Z_plane,
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.12, color='black', showlegend=False
        ))

    # Add dropdown menu for track selection
    buttons = [
        dict(
            label="Show All Tracks",
            method="update",
            args=[{"visible": [True] * len(fig.data)},  # Show all traces
                  {"title": "All Tracks Visible"}]
        )
    ]

    # Add a button for each track
    for i in range(len(pred_tracks)):
        visibility = [False] * len(fig.data)
        visibility[-1] = True  # Unassigned hits always visible
        visibility[i] = True  # Show only the selected track
        buttons.append(
            dict(
                label=f"Show Track {i}",
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Track {i} Visible"}]
            )
        )

    # Update layout with dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        scene=dict(
            xaxis=dict(title='X', range=x_lim),  # Imposta i limiti fissi per X
            yaxis=dict(title='GTK Station'),
            zaxis=dict(title='Y', range=y_lim),  # Imposta i limiti fissi per Y
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Interactive Predicted Tracks"
    )

    pio.write_html(fig, file=save_path, auto_open=False, include_plotlyjs='cdn')
    if show:
        fig.show()
    return fig