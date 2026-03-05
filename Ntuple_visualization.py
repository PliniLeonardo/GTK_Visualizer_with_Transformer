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
    data = read_root_file(config)
        
    # Build input tensor for visualization
    features_tensor_in_time_window, predicted_tracks_indexes , features_tensor = build_input(data)

    # Visualize the data on the GTK plane
    plot_gtk_hits_from_tensor(features_tensor_in_time_window, config['plot_folder_path'])
    gtk_dfs = split_hits_by_gtk(features_tensor_in_time_window, config['dataframe_path'])

    # Visualize the data in 3D interactive plot
    plot_3d_interactive_develop(
        predicted_tracks_indexes,
        features_tensor_in_time_window,
        features_tensor,
        data ,
        config,
        save_path=f"{config['plot_folder_path']}/Interactive_plot_tracks.html",
    )

    # Plot gif 
    if config['if_gif']:
        plot_3d_gif(predicted_tracks_indexes,
                    features_tensor,
                    config,
                    save_path=f"{config['plot_folder_path']}/gtk_3d.gif",)



if __name__ == "__main__":
    main()

