import streamlit as st
import yaml
import subprocess

st.title("Select time interval and visualizer")

config_file = "config_file_Ntuple.yaml"  # Modifica se necessario

# Load configuration file
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Input for time window
time_min = st.number_input("Min Time", value=config['time_window_min'])
time_max = st.number_input("Max Time", value=config['time_window_max'])

# Visualizer selection
visualizer = st.selectbox("Select the visualization", ["GTK", "Combined"], index=0 if config.get('visualizer', 'GTK') == 'GTK' else 1)

if st.button("Generate visualization"):
    # Update configuration file
    config['time_window_min'] = time_min
    config['time_window_max'] = time_max
    config['visualizer'] = visualizer
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    # Lancia lo script principale
    subprocess.run(["python3", "Ntuple_visualization.py", config_file])
    st.success("Updated!")