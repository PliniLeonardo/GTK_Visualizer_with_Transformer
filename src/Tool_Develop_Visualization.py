import uproot
import numpy as np
import torch
import os 
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from ipywidgets import interact, Text
from IPython.display import display

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

    z_mapping =  {79575: 0, 79625: 1, 86820: 2, 102400: 3}
    features_tensor[:, 2] = torch.tensor([z_mapping.get(int(val), val) for val in features_tensor[:, 2].numpy()])
    features_tensor_in_time_window[:, 2] = torch.tensor([z_mapping.get(int(val), val) for val in features_tensor_in_time_window[:, 2].numpy()])

    predicted_tracks_indexes = filter_predicted_tracks(predicted_tracks_indexes, mask.numpy())
    features_tensor_in_time_window = torch.tensor(features_tensor_in_time_window, dtype=torch.float32)
    
    return features_tensor_in_time_window, predicted_tracks_indexes, features_tensor


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


def plot_3d_interactive_develop(pred_tracks, 
                                features_tensor_in_time_window, 
                                features_tensor,
                                save_path,
                                marker_size=4, line_width=3, show=True):
    """
    Interactive 3D Plotly plot of predicted tracks. Unassigned hits drawn as black points.
    Args:
        pred_tracks: list of tracks (each track = list/array of hit indices)
        features_tensor_in_time_window: torch.Tensor or numpy array with columns [x, y, z_station, (t)] of hits in time window
        features_tensor: torch.Tensor or numpy array with columns [x, y, z_station, (t)] of original hits (not filtered by time window) to match track indices 
        save_path: Path to save the interactive HTML plot
        marker_size: Size of the markers for hits
        line_width: Width of the lines connecting hits in tracks
        show: Whether to display the plot in the browser
    """

    # Convert tensors to numpy arrays for easier manipulation
    features_in_time_window = features_tensor_in_time_window.numpy() if hasattr(features_tensor_in_time_window, "numpy") else np.asarray(features_tensor_in_time_window)
    features = features_tensor.numpy() if hasattr(features_tensor, "numpy") else np.asarray(features_tensor)

    # Extract columns for hits in time window
    x_in_time = features_in_time_window[:, 0]
    y_in_time = features_in_time_window[:, 1]
    z_station_in_time = features_in_time_window[:, 2].astype(int)

    # Extract columns for all hits
    x = features[:, 0]
    y = features[:, 1]
    z_station = features[:, 2].astype(int)

    fig = go.Figure()
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'grey', 'pink']

    # 1. Plot all hits in the time window as black points
    fig.add_trace(go.Scatter3d(
        x=x_in_time, y=z_station_in_time, z=y_in_time,
        mode='markers',
        marker=dict(size=marker_size, color='black'),
        name='Hits in time window',
        visible=True  # Always visible
    ))

    # 2. Plot tracks by connecting hits
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

    # 3. Define station planes for visualization
    x_lim = (-30.4, 30.4)  # Fixed limits for X
    y_lim = (-13.5, 13.5)  # Fixed limits for Y
    X_plane = [x_lim[0], x_lim[1], x_lim[1], x_lim[0]]
    Z_plane = [y_lim[0], y_lim[0], y_lim[1], y_lim[1]]
    for s in sorted(np.unique(z_station)):
        fig.add_trace(go.Mesh3d(
            x=X_plane, y=[s] * 4, z=Z_plane,
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.12, color='black', showlegend=False
        ))

    # 4. Add dropdown menu for track selection
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
        visibility = [True] * len(fig.data)
        # Hide all tracks except the selected one
        for j in range(len(pred_tracks)):
            if j != i:
                visibility[j + 1] = False  # +1 to skip the hits in time window
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
            xaxis=dict(title='X', range=x_lim),  # Fixed limits for X
            yaxis=dict(title='GTK Station'),
            zaxis=dict(title='Y', range=y_lim),  # Fixed limits for Y
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Interactive predicted tracks"
    )

    # 5. Save the plot as an HTML file and optionally display it
    pio.write_html(fig, file=save_path, auto_open=False, include_plotlyjs='cdn')
    if show:
        fig.show()
    return fig