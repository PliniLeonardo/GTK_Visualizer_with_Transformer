import uproot
import numpy as np
import torch
import os 
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import uproot_custom as ac
from IPython.display import display
import matplotlib.cm as cm 
import matplotlib.colors as mcolors

def read_root_file(config):
    """
    Reads data from a ROOT file and returns a dictionary with the specified keys.
    """

    required_keys = ['root_file', 'event_number', 'tree_path', 'x_key', 'y_key', 'z_key', 'time_key', 'KTAG_key', 'predicted_tracks_indexes',
                     'candidate_x_key', 'candidate_y_key', 'candidate_MomentumX_key', 'candidate_MomentumY_key', 'candidate_MomentumZ_key', 'candidate_time',
                     'straw_x1_key', 'straw_y1_key', 'straw_x_slope_key', 'straw_y_slope_key']
    # Give error if any required key is missing
    try:
        root_file, event_number, tree_path, x_key, y_key, z_key, time_key, ktag_time_key, predicted_tracks_indexes, candidate_x_key, candidate_y_key, candidate_MomentumX_key, candidate_MomentumY_key, candidate_MomentumZ_key, candidate_time_key, straw_x1_key, straw_y1_key, straw_x_slope_key, straw_y_slope_key = (
            config[key] for key in required_keys
        )
    except KeyError as e:
        raise ValueError(f"Missing required key in config: {e.args[0]}")

    ac.AsCustom.target_branches.add("/SlimReco:GigaTracker/fCandidates/fCandidates.fPositionX")
    ac.AsCustom.target_branches.add("/SlimReco:GigaTracker/fCandidates/fCandidates.fPositionY")
    with uproot.open(root_file) as f:
        candidate_x = f["SlimReco;1/GigaTracker/fCandidates/fCandidates.fPositionX"].array()[event_number]
        candidate_y = f["SlimReco;1/GigaTracker/fCandidates/fCandidates.fPositionY"].array()[event_number]

    with uproot.open(root_file) as f:
        predicted_tracks_indexes = f[predicted_tracks_indexes].array(library='np')[event_number]
        predicted_tracks_indexes = np.array([np.array(stl_vector) for stl_vector in predicted_tracks_indexes], dtype=object)


        # tree = f[tree_path]
        ac.AsCustom.target_branches.add(candidate_x_key)
        ac.AsCustom.target_branches.add(candidate_y_key)
        data = {
            'x': f[x_key].array(library='np')[event_number],
            'y': f[y_key].array(library='np')[event_number],
            'z': f[z_key].array(library='np')[event_number],
            'time': f[time_key].array(library='np')[event_number],
            'ktag_time': f[ktag_time_key].array(library='np')[event_number],
            'predicted_tracks_indexes': predicted_tracks_indexes,
            'candidate_x': candidate_x,
            'candidate_y': candidate_y, 
            'candidate_MomentumX': f[candidate_MomentumX_key].array(library='np')[event_number],
            'candidate_MomentumY': f[candidate_MomentumY_key].array(library='np')[event_number],
            'candidate_MomentumZ': f[candidate_MomentumZ_key].array(library='np')[event_number],
            'candidate_time': f[candidate_time_key].array(library='np')[event_number],
            'straw_x1': f[straw_x1_key].array(library='np')[event_number],
            'straw_y1': f[straw_y1_key].array(library='np')[event_number],
            'straw_x_slope': f[straw_x_slope_key].array(library='np')[event_number],
            'straw_y_slope': f[straw_y_slope_key].array(library='np')[event_number],
            'time_window_min': config['time_window_min'],
            'time_window_max': config['time_window_max']
            
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

def build_input (data):
    """
    Builds input tensor for the develop visualization from the provided data.
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :param time: time values
    :param ktag_time: ktag time values
    :param time_window: time window for filtering
    :param predicted_tracks_indexes: indexes of predicted tracks
    """
    x = data['x']
    y = data['y']
    z = data['z']
    time = data['time']
    ktag_time = data['ktag_time']
    start = data['time_window_min']
    end = data['time_window_max']
    predicted_tracks_indexes = data['predicted_tracks_indexes']

    time = time - ktag_time *24.95/256 # time -ktag time
    features = np.stack((x, y, z, time), axis=-1)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    mask = (features_tensor[:, 3] >= start) & (features_tensor[:, 3] <= end)
    features_tensor_in_time_window = features_tensor[mask]

    z_mapping =  {79575: 0.8 , 79625: 1, 86820: 1.6, 102400: 3}
    features_tensor[:, 2] = torch.tensor([z_mapping.get(int(val), val) for val in features_tensor[:, 2].numpy()])
    features_tensor_in_time_window[:, 2] = torch.tensor([z_mapping.get(int(val), val) for val in features_tensor_in_time_window[:, 2].numpy()])

    predicted_tracks_indexes_in_time = filter_predicted_tracks(predicted_tracks_indexes, mask.numpy())
    features_tensor_in_time_window = features_tensor_in_time_window.float()
    
    return features_tensor_in_time_window, predicted_tracks_indexes_in_time, features_tensor


def plot_gtk_hits_from_tensor(features_tensor, plot_folder_path):
    """
    Plots the hits on 4 different GTK planes using features_tensor, highlighting overlapping hits
    and positioning times around the hits to avoid overlap.
    """
    features = features_tensor.numpy()
    x = features[:, 0]
    y = features[:, 1]
    gtk_stations = features[:, 2].astype(float)
    t = features[:, 3]

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    axs = axs.flatten()

    x_lim = (-30.4, 30.4)
    y_lim = (-13.5, 13.5)
    markers = ['o', 'v', '<', '>', '^', 's', 'P', 'X', 'H', 'd'] * 20

    gtk_values = [0.8, 1, 1.6, 3]
    for idx, gtk in enumerate(gtk_values):
        gtk_hits = features[np.isclose(gtk_stations, gtk)]
        for i, hit in enumerate(gtk_hits):
            distances = np.sqrt((gtk_hits[:, 0] - hit[0])**2 + (gtk_hits[:, 1] - hit[1])**2)
            close_hits = distances <= 1

            if close_hits.sum() > 1:
                axs[idx].scatter(hit[0], hit[1],
                                 c='red',
                                 marker=markers[i % len(markers)],
                                 s=100, alpha=0.6, edgecolors='black')
            else:
                axs[idx].scatter(hit[0], hit[1],
                                 c='blue',
                                 marker=markers[i % len(markers)],
                                 s=100, edgecolors='black')

            nearby_hits = gtk_hits[close_hits]
            offsets = [(0.4, 0.4), (-0.4, -0.4), (-0.4, 0.4), (0.4, -0.4)]
            for j, nearby_hit in enumerate(nearby_hits):
                if j < len(offsets):
                    dx, dy = offsets[j]
                    axs[idx].text(nearby_hit[0] + dx, nearby_hit[1] + dy, round(nearby_hit[3], 2), fontsize=10)

        axs[idx].set_xlabel('x')
        axs[idx].set_ylabel('y')
        axs[idx].set_title(f'GTK{idx}', pad=20)
        axs[idx].grid()
        axs[idx].set_xlim(x_lim)
        axs[idx].set_ylim(y_lim)

    plt.tight_layout()
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)
    fig.savefig(os.path.join(plot_folder_path, 'GTK_hits_visualization.png'), dpi=300)
    plt.close(fig)

def split_hits_by_gtk(features_tensor, dataframe_path):
    """
    Splits the hits in features_tensor into 4 DataFrames, one for each GTK station.
    """
    features = features_tensor.numpy()
    x = features[:, 0]
    y = features[:, 1]
    z = features[:, 2]
    time = features[:, 3]
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'time': time})

    gtk_values = [0.8, 1, 1.6, 3]
    gtk_dfs = {}
    for i, gtk in enumerate(gtk_values):
        gtk_dfs[f'GTK{i}'] = df[np.isclose(df['z'], gtk)].reset_index(drop=True)

    if not os.path.exists(dataframe_path):
        os.makedirs(dataframe_path)
    for gtk, gtk_df in gtk_dfs.items():
        gtk_df.to_csv(os.path.join(dataframe_path, f'{gtk}_hits.csv'), index=False)
    
    return gtk_dfs


def add_filled_disk(fig, radius, y_position, color):
    """
    Adds a solid 3D disk to a Plotly figure using a parametric surface and draws its center axes.
    """
    # 1. Create a polar grid (r from 0 to radius, theta from 0 to 2pi)
    r = np.linspace(0, radius, 2)
    theta = np.linspace(0, 2*np.pi, 60)
    r_grid, theta_grid = np.meshgrid(r, theta)
    
    # 2. Convert Polar coordinates to Cartesian (X, Y, Z)
    x = r_grid * np.cos(theta_grid)
    z = r_grid * np.sin(theta_grid)
    y = np.full_like(x, y_position)
    
    # 3. Add to figure as a Surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.4,
        showlegend=False
    ))

    # 4. Add two orthogonal lines (diameters) through the center
    fig.add_trace(go.Scatter3d(
        x=[-radius, radius], y=[y_position, y_position], z=[0, 0],
        mode='lines',
        line=dict(color='blue', width=6, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[y_position, y_position], z=[-radius, radius],
        mode='lines',
        line=dict(color='blue', width=6, dash='dash'),
        showlegend=False
    ))
    # 5. Add border (contour) of the disk
    theta_border = np.linspace(0, 2*np.pi, 100)
    x_border = radius * np.cos(theta_border)
    z_border = radius * np.sin(theta_border)
    y_border = np.full_like(x_border, y_position)
    fig.add_trace(go.Scatter3d(
        x=x_border, y=y_border, z=z_border,
        mode='lines',
        line=dict(color='black', width=4),
        showlegend=False
    ))

def update_gtk_layout(fig):
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', range=[-40, 40]),
                yaxis=dict(
                    title='',
                    tickmode='array',
                    tickvals=[0.8 , 1, 1.6, 3],
                    ticktext=['GTK0', 'GTK1', 'GTK2', 'GTK3'],
                    range=[-1, 4]
                ),
                zaxis=dict(title='Y', range=[-40, 40])
            )
        )
        


def plot_3d_interactive_develop(pred_tracks, 
                                features_tensor_in_time_window, 
                                features_tensor,
                                data,
                                config,
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

    # COLORS
    colorscale = 'rainbow'
    cmin = -10
    cmax = 10
    norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
    cmap = cm.get_cmap(colorscale)

    # Convert tensors to numpy arrays for easier manipulation
    features_in_time_window = features_tensor_in_time_window.numpy() if hasattr(features_tensor_in_time_window, "numpy") else np.asarray(features_tensor_in_time_window)
    features = features_tensor.numpy() if hasattr(features_tensor, "numpy") else np.asarray(features_tensor)

    # Extract columns for hits in time window
    x_in_time = features_in_time_window[:, 0]
    y_in_time = features_in_time_window[:, 1]
    z_station_in_time = features_in_time_window[:, 2].astype(float)

    # Extract columns for all hits
    x = features[:, 0]
    y = features[:, 1]
    z_station = features[:, 2].astype(float)
    times = features[:, 3] #- data['ktag_time'] *24.95/256 

    fig = go.Figure()
    
    color_rgb = cmap(norm(times))
    color_hex = np.array([mcolors.to_hex(c) for c in color_rgb])
    
    # 1. Plot all hits in the time window as black points
    fig.add_trace(go.Scatter3d(
        x=x_in_time, y= z_station_in_time, z=y_in_time, 
        mode='markers',
        marker=dict(
            size=marker_size,
            color='black'
        ),
        name = 'Hits in Time Window',
        showlegend= True,  
    ))

    # 2. Plot tracks by connecting hits
    for i, track in enumerate(pred_tracks):
        idx = np.asarray(track, dtype=int)
        if idx.size == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=x[idx], y= z_station[idx] , z=y[idx],
            mode='lines+markers',
            line=dict(width=line_width, color= color_hex[idx]),
            marker=dict(size=marker_size, color= color_hex[idx]),
            showlegend=False,
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
            opacity=0.1, color='black', showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[X_plane[0], X_plane[1], X_plane[2], X_plane[3], X_plane[0]],
            y=[s, s, s, s, s],
            z=[Z_plane[0], Z_plane[1], Z_plane[2], Z_plane[3], Z_plane[0]],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))

    # Add invisible scatter to plot colorbar on the left
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],  
        mode='markers',
        marker=dict(
            size=0.1,
            color=[-10, 0, 10],  
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title='Time (ns)',
                thickness=20,
                len=0.5,
                x=0.02  
            )
        ),
        showlegend=False
    ))
    
    update_gtk_layout(fig)


    # 4. Candidates positions on GTK3
    candidates_time = data['candidate_time'] - data['ktag_time'] *24.95/256
    # select candidates in the time window
    mask_candidates = (candidates_time >= config['time_window_min']) & (candidates_time <= config['time_window_max'])
    dz = (11 - 3) * 10000  # distance between GTK3 and Straw1 is approximately 80 m= 80 0000 mm 
    color_rgb = cmap(norm(candidates_time))
    color_hex = np.array([mcolors.to_hex(c) for c in color_rgb])

    for i in range(len(data['candidate_x'])):
        if mask_candidates[i]: 
            # Take candidate position on GTK3
            x0 = data['candidate_x'][i][-1] 
            y0 = data['candidate_y'][i][-1] 
            slope_x = data['candidate_MomentumX'][i] / data['candidate_MomentumZ'][i] 
            slope_y = data['candidate_MomentumY'][i] / data['candidate_MomentumZ'][i] 
            x1 = x0 + -slope_x * dz # THANKS MATT: pay attention to the coordinate system! positives x means that the beam goes to the left
            y1 = y0 + slope_y * dz

            # Marker on GTK3 to represent the candidate position 
            fig.add_trace(go.Scatter3d(
                x=[x0], y=[3], z=[y0],
                mode='markers',
                marker = dict(size= 4, color= color_hex[i], symbol="diamond"),
                showlegend= False
            ))

            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[3, 11], z=[y0, y1],
                mode='lines',
                line=dict(width =line_width, color= color_hex[i]),
                name=f"Candidate {i}",
                showlegend= True
            ))
           



    if config["visualizer"] == "Combined":
        # # STRAW
        # Straw planes
        add_filled_disk(fig, radius=1050, y_position=11, color='blue')
        add_filled_disk(fig, radius=1050, y_position=12, color='blue')

        scale = 1
        dz_reale = 10000
        # scale the images if necessary
        x1_graph = data['straw_x1'] / scale
        z1_graph = data['straw_y1'] / scale 

        # Slope 
        x_slope_graph = (data['straw_x_slope'] * dz_reale) / scale
        z_slope_graph = (data['straw_y_slope'] * dz_reale) / scale

        for i in range(len(x1_graph)):
            y_vals = np.array([3,4,5,6,7,8,9,10,11, 12]) # Bastano due punti per una linea retta
            x_vals = x1_graph[i] + x_slope_graph[i] * (y_vals - 11)
            z_vals = z1_graph[i] + z_slope_graph[i] * (y_vals - 11)
            
            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines',
                line=dict(width=4, color='black', dash='dash'),
                showlegend=False
            ))
            # at the intersection  between segment and the plane y=11 and y=12 add the points
            y_intersection_11 = 11
            x_intersection_11 = x1_graph[i] + x_slope_graph[i] * (y_intersection_11 - 11)
            z_intersection_11 = z1_graph[i] + z_slope_graph[i] * (y_intersection_11 - 11)

            y_intersection_12 = 12
            x_intersection_12 = x1_graph[i] + x_slope_graph[i] * (y_intersection_12 - 11)
            z_intersection_12 = z1_graph[i] + z_slope_graph[i] * (y_intersection_12 - 11)

            fig.add_trace(go.Scatter3d(
                x=[x_intersection_11], y=[y_intersection_11], z=[z_intersection_11],
                mode='markers',
                marker=dict(size=4, color='black'),
                showlegend = False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[x_intersection_12], y=[y_intersection_12], z=[z_intersection_12],
                mode='markers',
                marker=dict(size=4, color='black'),
                showlegend= False
            ))

        # 4. Update layout for better visualization
        fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-1200, 1200]),  # o più stretto se vuoi
            yaxis=dict(
                title='Z',
                tickmode='array',
                tickvals=[0, 1, 2, 3, 11, 12],
                ticktext=['GTK0', 'GTK1', 'GTK2', 'GTK3', 'Straw1', 'Straw2'],
                range=[-1, 13]  # così vedi sia GTK che Straw
            ),
            zaxis=dict(title='Y', range=[-1200, 1200])
        )
    )
            


    # 5. Save the plot as an HTML file and optionally display it
    pio.write_html(fig, file=save_path, auto_open=False, include_plotlyjs='cdn')
    if show:
        fig.show()
    return fig

