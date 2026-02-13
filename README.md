Visualize the GTK station tracks and hits in [-10, +10] nsec window.


# Installation 

Execute the script ```./setup_env.sh``` to create the virtual environment.
Activate the environment  ```source Visualization_environment/bin/activate```



# Commands to run Visualization with Ntuple from Reconstruction
1. Insert your root file in the folder data_to_visualize and modify the path in the config_file_Ntuple.yaml (key = root_file:) accordingly
2. Use the config_file_Ntuple.yaml to modify the parameters
3. Run the command ```streamlit run visualization_app.py``` to open the streamlit app in your browser. Insert the time window and select the visualizer you want to use ("GTK" to see only the GTK and "Combined" to see both GTK and Straw), then click on "Generate visualization" to update the plots.
<!-- 3. Run the command ``` python Ntuple_Visualization.py config_file_Ntuple.yaml``` . -->

You will find the following outputs in the folder plots
* Interactive_plot_tracks.html: interactive 3D plot of the tracks on the GTK stations and Straw by using the argument visualizer: "Combined" in the config file. To visualize only the GTK stations, use visualizer: "GTK" in the config file.
* GTK_hits_visualization.png: 4 panels visualization that displays also the time of each hit. Very near hits are represented in red
* GTK0_hits.csv, GTK1_hits.csv, GTK2_hits.csv, GTK3_hits.csv: csv files with x,y and time for each hit divided by GTK station



