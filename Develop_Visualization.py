import sys
import yaml
from src.Tool_Develop_Visualization import *

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

    data = read_root_file(root_file, tree_path, x_key, y_key, z_key, time_key, predicted_tracks_indexes )

    # Build input tensor for the develop visualization
    features_tensor_in_time_window, predicted_tracks_indexes , features_tensor= build_input_for_develop(
        data['x'], data['y'], data['z'], data['time'], time_window,  data['predicted_tracks_indexes']
    )

    # Visualize the data
    plot_folder_path = config['plot_folder_path']
    plot_gtk_hits_from_tensor(features_tensor, plot_folder_path)


    print (" DAJEEE ")

if __name__ == "__main__":
    main()