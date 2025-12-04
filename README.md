Visualize the GTK station tracks and hits


# Installation 

Execute the script ```./setup_env.sh``` to create the virtual environment.
Activate the environment  ```source VisualizationGTK_environment/bin/activate```

# Commands
1. Insert your root file in the folder data_to_visualize
2. Use the config_file.yaml to modify the parameters
3. Run the command ``` python Transformer_Visualization.py config_file.yaml``` .

Notice that
- this implementation needs the KTAG Time.
- the hits are not clustered and the pattern recognition method is applied on top of unclustered hits. 