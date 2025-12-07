Visualize the GTK station tracks and hits


# Installation 

Execute the script ```./setup_env.sh``` to create the virtual environment.
Activate the environment  ```source VisualizationGTK_environment/bin/activate```

# Commands to run Transformer Visualization
1. Insert your root file in the folder data_to_visualize
2. Use the config_file.yaml to modify the parameters
3. Run the command ``` python Transformer_Visualization.py config_file_Transformer.yaml``` .

You will find the following outputs in the folder plots
* Interactive_plot_tracks.html: interactive 3D plot of the tracks
* GTK_hits_visualization.png: 4 panels visualization that displays also the time of each hit. Very near hits are represented in red
* GTK0_hits.csv, GTK1_hits.csv, GTK2_hits.csv, GTK3_hits.csv: csv files with x,y and time for each hit divided by GTK station

Notice that
- this implementation needs the KTAG Time.
- the hits are not clustered and the pattern recognition method is applied on top of unclustered hits. 


# Commands to run Develop Visualization
1. Insert your root file in the folder data_to_visualize
2. Use the config_file_Develop.yaml to modify the parameters
3. Run the command ``` python Develop_Visualization.py config_file_Develop.yaml``` .


# Commands to run Visualization with Ntuple from Reconstruction
1. Insert your root file in the folder data_to_visualize
2. Use the config_file_Ntuple.yaml to modify the parameters
3. Run the command ``` python Ntuple_Visualization.py config_file_Ntuple.yaml``` .