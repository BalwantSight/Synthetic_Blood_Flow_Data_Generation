import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Physical and biomedical constants
g = 9.81  # Gravitational acceleration in m/s²
rho_blood = 1060  # Blood density in kg/m³
viscosity_blood_normal = 0.0035  # Normal blood viscosity in Pa·s
normal_hematocrit = 0.45  # Normal hematocrit (45%)

# Physiological parameters for blood vessels in hypertension
vessel_params = {
    'artery': {
        'diameter_range': (0.0005, 0.010),  # 0.5-10 mm
        'length_range': (0.1, 0.5),  # 10-50 cm
        'pressure_systolic': (130000, 180000),
        'pressure_diastolic': (80000, 120000)  # High diastolic pressure
    },
}

def calculate_blood_viscosity(hematocrit):
    """Calculates blood viscosity as a function of hematocrit."""
    return viscosity_blood_normal * (1 + 3.5 * hematocrit)

def generate_synthetic_blood_flow_data(n_samples=15000, vessel_type='artery', pathological_state='hypertension', file_name='synthetic_blood_flow_data.csv'):
    """Generates synthetic blood flow data considering physiology and pathological conditions."""
    params = vessel_params[vessel_type]

    # Vessel diameters (m)
    diameters = np.random.lognormal(mean=np.log((params['diameter_range'][0] + params['diameter_range'][1]) / 2), sigma=0.25, size=n_samples)
    diameters = np.clip(diameters, params['diameter_range'][0], params['diameter_range'][1])

    # Vessel lengths
    lengths = np.random.uniform(params['length_range'][0], params['length_range'][1], n_samples)

    # Initial and final pressures (Pa) for hypertension
    pressures_initial = np.random.normal(loc=(params['pressure_systolic'][0] + params['pressure_systolic'][1]) / 2, scale=(params['pressure_systolic'][1] - params['pressure_systolic'][0]) / 6, size=n_samples)
    pressure_drop = (0.06 * (1 - (diameters / params['diameter_range'][1])) * (1 - (lengths / params['length_range'][1]))) * pressures_initial
    pressures_final = pressures_initial - pressure_drop

    # Heart rate
    heart_rate = np.random.normal(loc=80, scale=10, size=n_samples)
    if pathological_state == 'tachycardia':
        heart_rate = np.random.uniform(100, 180, n_samples)
    elif pathological_state == 'bradycardia':
        heart_rate = np.random.uniform(40, 60, n_samples)

    # Hematocrit and viscosity
    hematocrit = np.random.normal(0.48, 0.05, n_samples)
    blood_viscosity = calculate_blood_viscosity(hematocrit)

    # Calculate volumetric flow using Poiseuille's law considering increased vascular resistance
    radii = diameters / 2
    pressure_diff = pressures_initial - pressures_final
    resistance_factor = 1.5  # Increased resistance factor for hypertension
    flow_rates = (pressure_diff * np.pi * radii**4) / (8 * blood_viscosity * lengths)

    # Blood flow velocity
    velocities = flow_rates / (np.pi * radii**2)

    # Blood composition (hemoglobin)
    sex = np.random.choice(['male', 'female'], n_samples)
    hemoglobin = np.where(sex == 'male', np.random.normal(16, 1, n_samples), np.random.normal(14, 1, n_samples))

    # Convert sex to numeric
    sex_numeric = np.where(sex == 'male', 1, 0)

    # Age of subjects
    age = np.random.randint(18, 90, size=n_samples)

    # Normalize flow and velocity data
    scaler = MinMaxScaler()
    normalized_flow = scaler.fit_transform(flow_rates.reshape(-1, 1))
    normalized_velocity = scaler.fit_transform(velocities.reshape(-1, 1))

    # Create a DataFrame with the generated data
    data = {
        'Diameter (m)': diameters,
        'Initial Pressure (Pa)': pressures_initial,
        'Final Pressure (Pa)': pressures_final,
        'Length (m)': lengths,
        'Heart Rate (beats/min)': heart_rate,
        'Hematocrit': hematocrit,
        'Viscosity (Pa·s)': blood_viscosity,
        'Flow (m³/s)': normalized_flow.flatten(),
        'Velocity (m/s)': normalized_velocity.flatten(),
        'Hemoglobin (g/dL)': hemoglobin,
        'Age (years)': age,
        'Sex': sex_numeric  # Numeric representation of sex
    }

    df = pd.DataFrame(data)

    # Ensure data/ directory exists, create if not
    os.makedirs('./data', exist_ok=True)

    # Save data to CSV file
    df.to_csv(f'./data/{file_name}', index=False)
    print(f"Generated {n_samples} samples and saved to {file_name}")

    # Visualize flow distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Flow (m³/s)'], bins=50, kde=True)
    plt.title('Blood Flow Distribution in Hypertension')
    plt.xlabel('Flow (m³/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Visualize vessel diameter distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Diameter (m)'], bins=50, kde=True)
    plt.title('Vessel Diameter Distribution')
    plt.xlabel('Diameter (m)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Visualize relationship between heart rate and flow
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df['Heart Rate (beats/min)'], y=df['Flow (m³/s)'], alpha=0.5)
    plt.title('Blood Flow vs Heart Rate in Hypertension')
    plt.xlabel('Heart Rate (beats/min)')
    plt.ylabel('Flow (m³/s)')
    plt.grid(True)
    plt.show()

    # Visualize relationship between hematocrit and viscosity
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df['Hematocrit'], y=df['Viscosity (Pa·s)'], alpha=0.5)
    plt.title('Viscosity vs Hematocrit in Hypertension')
    plt.xlabel('Hematocrit')
    plt.ylabel('Viscosity (Pa·s)')
    plt.grid(True)
    plt.show()

    # Create correlation heatmap
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
    plt.title('Correlation Heatmap between Variables')

    # Rotate x and y axis labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)

    plt.show()

    # Statistical analysis of the data
    stats_summary = df.describe()

    # Create graphical table of the statistical analysis
    plt.figure(figsize=(14, 10))  # Increase figure size
    stats_heatmap = sns.heatmap(stats_summary, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, square=True)
    plt.title('Statistical Analysis of the Data', fontsize=16)  # Increase font size for the title
    plt.xlabel('Variables', fontsize=12)  # Increase font size for labels
    plt.ylabel('Statistics', fontsize=12)  # Increase font size for labels

    # Rotate x and y axis labels in the statistical analysis
    stats_heatmap.set_xticklabels(stats_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
    stats_heatmap.set_yticklabels(stats_heatmap.get_yticklabels(), rotation=0, fontsize=10)

    plt.show()

if __name__ == "__main__":
    # Call the function to generate synthetic blood flow data
    generate_synthetic_blood_flow_data()
