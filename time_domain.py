import matplotlib.pyplot as plt
import pandas as pd

# read csv
#smooth surface
smooth_laminar_data = pd.read_csv('smooth_data_with_times.csv') #laminar flow data
smooth_turbulent_data = pd.read_csv('smooth_turbulent_with_times.csv') #turbulent flow data
#micro surface
microstructure_laminar_data = pd.read_csv('micro_data_with_times.csv') #laminar flow data
microstructure_turbulent_data = pd.read_csv('micro_turbulent_with_times.csv') #turbulent flow data

# time domain for smooth surface
#shows the temp vs voltage graphs (same as on the excel sheet)
for column in smooth_laminar_data.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(smooth_laminar_data.index, smooth_laminar_data[column], label=f"Smooth Surface {column}")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Temperature over Time for {column} - Smooth Surface Laminar Flow")
    plt.legend()
    plt.show()

# time domain for microstructure surface data
for column in microstructure_laminar_data.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(microstructure_laminar_data.index, microstructure_laminar_data[column], label=f"Microstructure Surface {column}")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Temperature over Time for {column} - Microstructure Surface Laminar Flow")
    plt.legend()
    plt.show()
