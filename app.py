import os
from data_generator import generate_synthetic_blood_flow_data
import subprocess

def main():
    # Generate synthetic data
    n_samples = 15000  # You can allow users to change this if desired
    file_name = 'synthetic_blood_flow_data.csv'
    generate_synthetic_blood_flow_data(n_samples=n_samples, file_name=file_name)

    # Run the dashboard
    # Here I assume the dashboard is a Python script that can be executed.
    subprocess.run(['python', 'dashboard.py'])

if __name__ == "__main__":
    main()
