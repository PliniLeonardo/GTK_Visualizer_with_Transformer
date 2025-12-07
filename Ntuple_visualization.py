import sys
import yaml
from src.Tool_Ntuple_Visualization import *

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python Transformer_Visualization.py <config_file.yaml>")
        sys.exit(1)

    # Load configuration file
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Read data from ROOT file
    root_file = config['root_file']
    tree_path = config['tree_path']
    x_key = config['x_key']
    y_key = config['y_key']
    z_key = config['z_key']
    time_key = config['time_key']
    time_window = (config['time_window_min'], config['time_window_max'])
    predicted_tracks_indexes = config['predicted_tracks_indexes']

    data = read_root_file(config)
        # root_file, tree_path, x_key, y_key, z_key, time_key, predicted_tracks_indexes )

    # Build input tensor for the develop visualization
    features_tensor_in_time_window, predicted_tracks_indexes , features_tensor= build_input(
        data['x'], data['y'], data['z'], data['time'], time_window,  data['predicted_tracks_indexes']
    )

    # Visualize the data
    plot_folder_path = config['plot_folder_path']
    plot_gtk_hits_from_tensor(features_tensor_in_time_window, plot_folder_path)


    gtk_dfs = split_hits_by_gtk(features_tensor_in_time_window, config['dataframe_path'])

    plot_3d_interactive_develop(
        predicted_tracks_indexes,
        features_tensor_in_time_window,
        features_tensor,
        save_path=f"{plot_folder_path}/Interactive_plot_tracks.html"
    )


    print (" DAJEEE ")

if __name__ == "__main__":
    main()