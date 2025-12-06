import sys
import yaml
from src.Tool_Transformer_Visualization import * 

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

    data = read_root_file(root_file, tree_path, x_key, y_key, z_key, time_key)

    # Build input tensor for the transformer
    features_tensor, pair_indices = build_input_for_transformer(
        data['x'], data['y'], data['z'], data['time'], time_window
    )

    # Load the transformer model
    model = transformer_model_load()

    # Make predictions with the model
    model_output = model(features_tensor, pair_indices)

    # Post-process predictions
    edge_dict = edge_dictionary_transformer(model_output, config['transformer_threshold'], pair_indices)
    pred_tracks = make_pred_tracks(edge_dict, features_tensor)

    # Visualize the data
    plot_folder_path = config['plot_folder_path']
    plot_gtk_hits_from_tensor(features_tensor, plot_folder_path)

    gtk_dfs = split_hits_by_gtk(features_tensor, config['dataframe_path'])

    plot_3d_interactive(
        pred_tracks,
        features_tensor,
        save_path=f"{plot_folder_path}/Interactive_plot_tracks.html"
    )

    print(f" ------- Terminated successfully -------")
    print(f" Plots saved in folder: {plot_folder_path} in time interval {time_window} ns")



if __name__ == "__main__":
    main()

