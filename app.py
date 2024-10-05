import os
from data_generator import generate_synthetic_blood_flow_data
import subprocess

def main():
    # Generar datos sintéticos
    n_samples = 15000  # Puedes permitir que los usuarios cambien esto si lo deseas
    file_name = 'synthetic_blood_flow_data.csv'
    generate_synthetic_blood_flow_data(n_samples=n_samples, file_name=file_name)

    # Ejecutar el dashboard
    # Aquí asumo que el dashboard es un script de Python que puede ser ejecutado.
    subprocess.run(['python', 'dashboard.py'])

if __name__ == "__main__":
    main()
