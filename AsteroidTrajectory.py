import time

from Newtonian import SolarSystem
from Newtonian import CelestialBodyData

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os


EARTH_RADIUS = 6371000
data_directory = "SimulationData"
AU = np.float64(1.496e+11)  # Astronomical Unit in meters


# Section for comparing integrator performance
def plot_energy_and_angular_momentum_integrators(fileNames, step_interval=10):
    # Plot for Energy Drift
    fig_energy, ax_energy = plt.subplots(figsize=(15, 5))

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")
        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            ke_history = np.array(simulation_data["kinetic_energy"])[::step_interval]
            pe_history = np.array(simulation_data["potential_energy"])[::step_interval]
            angular_momentum = np.array(simulation_data["angular_momentum"])[::step_interval]

            # Calculate total energy and energy drift
            total_energy = ke_history + pe_history
            initial_total_energy = total_energy[0]
            energy_drift_percent = (total_energy - initial_total_energy) / initial_total_energy * 100

            # Plot Energy Drift
            ax_energy.plot(energy_drift_percent, label=f"{fileName} Energy Drift (%)")

            # Calculate min, max, and final energy drift
            max_drift = np.max(np.abs(energy_drift_percent))
            final_drift = energy_drift_percent[-1]

            print(
                f"{fileName}. Max Energy Drift: {max_drift:.5e}%, Final Energy Drift: {final_drift:.5e}%"
            )

    ax_energy.set_title("Energy Drift (%)", fontsize=20)
    ax_energy.set_xlabel("Step", fontsize=16)
    ax_energy.set_ylabel("Energy Drift (%)", fontsize=16)
    ax_energy.tick_params(axis='both', which='major', labelsize=14)
    ax_energy.legend(fontsize=14)
    plt.show()

    # Plot for Angular Momentum
    fig_angular, axs_angular = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    final_angular_momentums = {}

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")
        with open(file_path, "r") as file:
            simulation_data = json.load(file)
            angular_momentum = np.array(simulation_data["angular_momentum"])[::step_interval]

            # Plot each component of Angular Momentum
            for i in range(3):
                axs_angular[i].plot(angular_momentum[:, i], label=f"{fileName} Angular Momentum {['X', 'Y', 'Z'][i]}")

            final_angular_momentums[fileName] = angular_momentum[-1]

    for ax in axs_angular:
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

    axs_angular[0].set_title("Angular Momentum X Component", fontsize=20)
    axs_angular[1].set_title("Angular Momentum Y Component", fontsize=20)
    axs_angular[2].set_title("Angular Momentum Z Component", fontsize=20)
    axs_angular[2].set_xlabel("Step", fontsize=16)
    plt.show()


def plot_angular_momentum_z_drift(fileNames, step_interval=10):
    fig, ax = plt.subplots(figsize=(15, 5))

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")
        with open(file_path, "r") as file:
            simulation_data = json.load(file)
            angular_momentum = np.array(simulation_data["angular_momentum"])[::step_interval]

            # Extract the Z component of angular momentum
            angular_momentum_z = angular_momentum[:, 2]

            # Calculate the initial Z component of angular momentum
            initial_angular_momentum_z = angular_momentum_z[0]

            # Calculate the percentage drift in the Z component
            angular_momentum_z_drift_percent = (angular_momentum_z - initial_angular_momentum_z) / initial_angular_momentum_z * 100

            # Plot the percentage drift
            ax.plot(angular_momentum_z_drift_percent, label=f"{fileName} Angular Momentum Z Drift (%)")

            # Calculate and print the max drift
            max_drift = np.max(np.abs(angular_momentum_z_drift_percent))
            print(f"Max Angular Momentum Z Drift for {fileName}: {max_drift}%")

    ax.set_title("Angular Momentum Z Component Drift (%)", fontsize=20)
    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Drift (%)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)
    plt.show()


def compare_integrators_performance(fileNames):
    earth_movement_per_integrator = {}
    earth_final_positions = {}  # Store Earth's final positions for comparison
    distance_to_rk4 = {}
    angular_change_per_integrator = {}
    rk4_final_position = None
    rk4_angular_change = None
    rk4_earth_final_position = None

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            # Get Earth's and Ceres' trajectory
            earth_trajectory = np.array(simulation_data["trajectories"]["Earth"])
            ceres_trajectory = np.array(simulation_data["trajectories"]["asteroid Ceres: 9.39×10^20"])

            # Store Earth's final position for comparison
            earth_final_positions[fileName] = earth_trajectory[-1]

            # Calculate angular change
            initial_vector = ceres_trajectory[1] - ceres_trajectory[0]
            final_vector = ceres_trajectory[-1] - ceres_trajectory[-2]
            initial_vector_normalized = initial_vector / np.linalg.norm(initial_vector)
            final_vector_normalized = final_vector / np.linalg.norm(final_vector)
            angle = np.arccos(np.clip(np.dot(initial_vector_normalized, final_vector_normalized), -1.0, 1.0))
            angular_change_per_integrator[fileName] = np.degrees(angle)

            # Save the final positions and angular change for RK4 for comparison
            if "RK4" in fileName:
                rk4_final_position = ceres_trajectory[-1]
                rk4_earth_final_position = earth_final_positions[fileName]
                rk4_angular_change = angular_change_per_integrator[fileName]

    # Calculate distances to RK4 and angular change differences
    for fileName in fileNames:
        if "RK4" not in fileName:
            distance_to_rk4[fileName] = np.linalg.norm(earth_final_positions[fileName] - rk4_earth_final_position)
            angular_change_difference = angular_change_per_integrator[fileName] - rk4_angular_change
            print(f"Distance from {fileName} to RK4 at end: {distance_to_rk4[fileName]:.5e} meters")
            print(f"Angular change difference compared to RK4: {angular_change_difference:.5e} degrees")

    # Print Earth's movement during each simulation compared to RK4
    for fileName in fileNames:
        earth_movement_difference = np.linalg.norm(earth_final_positions[fileName] - rk4_earth_final_position)
        print(f"Earth's movement difference in {fileName} simulation compared to RK4: {earth_movement_difference:.5e} meters")


def simulate_integrator_comparison():
    # Define Ceres asteroid data
    ceres = CelestialBodyData("asteroid Ceres: 9.39×10^20", "green", 9.39e+20, 0, 0)

    dt = 0.01  # Time step in seconds
    SIMULATION_TIME = 3600 * 3  # Seconds, 3 hours simulated

    # Integrators to be used
    integrators = [SolarSystem.euler_update, SolarSystem.verlet_update, SolarSystem.rk4_update]

    for integrator_method in integrators:
        integrator_name = integrator_method.__name__.replace('_update', '')
        print(integrator_name)
        print(f"Running simulation for Ceres using {integrator_name} integrator")

        # Create a new solar system instance for each integrator
        solar_system = SolarSystem()
        solar_system.add_body_at_center(CelestialBodyData.get_earth_data())
        solar_system.add_body_relative_to_other(ceres, "Earth", EARTH_RADIUS / 1000 * 3, 10, [30, 0])

        # Bind the solar_system instance to the integrator method
        bound_integrator = integrator_method.__get__(solar_system, SolarSystem)

        start_time = time.time()

        # Run the simulation with the specified integrator
        solar_system.run_simulation(timeStep=dt, SIMULATION_TIME=SIMULATION_TIME, integrator=bound_integrator,
                                    fileName=f"Ceres_{integrator_name}")

        # End timer and calculate elapsed time
        elapsed_time = time.time() - start_time

        # Print simulation time for each integrator
        print(f"Simulation for {integrator_name} integrator completed in {elapsed_time:.2f} seconds")


def main_compare_integrators():
    simulate_integrator_comparison()

    # Plot energy and angular momentum for this integrator
    fileNames = [f"Ceres_{integrator_name}" for integrator_name in ["Verlet", "Euler", "RK4"]]

    plot_energy_and_angular_momentum_integrators(fileNames)
    plot_angular_momentum_z_drift(fileNames, step_interval=10)
    compare_integrators_performance(fileNames)

    plot_trajectories_with_ceres(fileNames, EARTH_RADIUS)


# Section for investigating the mass influence on the trajectories
def plot_energy_and_angular_momentum(fileName, step_interval=1000):
    file_path = os.path.join(data_directory, f"{fileName}.json")

    with open(file_path, "r") as file:
        simulation_data = json.load(file)

        ke_history = simulation_data["kinetic_energy"]
        pe_history = simulation_data["potential_energy"]
        angular_momentum = simulation_data["angular_momentum"]

        total_energy = [ke + pe for ke, pe in zip(ke_history, pe_history)]
        initial_total_energy = total_energy[0]
        energy_drift_percent = [(e - initial_total_energy) / initial_total_energy * 100 for e in total_energy]

        selected_steps = range(0, len(energy_drift_percent), step_interval)
        selected_energy_drift = [energy_drift_percent[i] for i in selected_steps]
        selected_angular_momentum = [angular_momentum[i] for i in selected_steps]

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Font size settings
        title_fontsize = 20
        label_fontsize = 16
        legend_fontsize = 14

        axs[0].plot(selected_steps, selected_energy_drift, label="Asteroids Energy Drift (%)")
        axs[0].set_title("Energy Drift (%)", fontsize=title_fontsize)
        axs[0].set_xlabel("Step", fontsize=label_fontsize)
        axs[0].set_ylabel("Energy Drift (%)", fontsize=label_fontsize)
        axs[0].legend(fontsize=legend_fontsize)
        axs[0].tick_params(axis='both', which='major', labelsize=label_fontsize)

        axs[1].plot(selected_steps, selected_angular_momentum, label="Asteroids Angular Momentum")
        axs[1].set_title("Angular Momentum", fontsize=title_fontsize)
        axs[1].set_xlabel("Step", fontsize=label_fontsize)
        axs[1].set_ylabel("Momentum", fontsize=label_fontsize)
        axs[1].legend(fontsize=legend_fontsize)
        axs[1].tick_params(axis='both', which='major', labelsize=label_fontsize)

        plt.show()


def analyze_asteroid_trajectories(fileNames):
    earth_initial_final_movement = {}
    asteroid_final_distances = {}
    angular_changes = {}
    smallest_asteroid_angular_change = None

    # Load the trajectory of the smallest asteroid (100 kg) for comparison
    smallest_asteroid_trajectory = None
    smallest_asteroid_file_path = os.path.join(data_directory, "Small asteroid: 100kg.json")
    with open(smallest_asteroid_file_path, "r") as file:
        simulation_data = json.load(file)
        smallest_asteroid_trajectory = np.array(simulation_data["trajectories"]["Small asteroid: 100kg"])

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            # Get Earth's and the asteroid's trajectory
            earth_trajectory = np.array(simulation_data["trajectories"]["Earth"])
            asteroid_trajectory = np.array(simulation_data["trajectories"][fileName])

            # Calculate Earth's movement (first minus last position)
            earth_movement = np.linalg.norm(earth_trajectory[-1] - earth_trajectory[0])
            earth_initial_final_movement[fileName] = earth_movement

            # Calculate the distance between this asteroid and the smallest asteroid at the last step
            distance_to_smallest_asteroid = np.linalg.norm(asteroid_trajectory[-1] - smallest_asteroid_trajectory[-1])
            asteroid_final_distances[fileName] = distance_to_smallest_asteroid

            # Calculate angular change
            initial_vector = asteroid_trajectory[1] - asteroid_trajectory[0]
            final_vector = asteroid_trajectory[-1] - asteroid_trajectory[-2]
            initial_vector_normalized = initial_vector / np.linalg.norm(initial_vector)
            final_vector_normalized = final_vector / np.linalg.norm(final_vector)
            angle = np.arccos(np.clip(np.dot(initial_vector_normalized, final_vector_normalized), -1.0, 1.0))
            angular_changes[fileName] = np.degrees(angle)

    # Extract angular change for the smallest asteroid
    smallest_asteroid_angular_change = angular_changes["Small asteroid: 100kg"]

    # Print Earth's total movement for each asteroid
    for fileName, movement in earth_initial_final_movement.items():
        print(f"Earth's movement during {fileName} simulation: {movement:.5e} meters")

    # Print the final distances between each asteroid and the smallest asteroid
    for fileName, distance in asteroid_final_distances.items():
        print(f"Distance between {fileName} and 'Small asteroid: 100kg' at end of simulation: {distance:.5e} meters")

    # Print the angular changes and their differences compared to the smallest asteroid
    for fileName, angle in angular_changes.items():
        angle_difference = angle - smallest_asteroid_angular_change
        print(f"The total angular change for {fileName} is {angle:.5e} degrees.")
        print(f"Angular change difference compared to 'Small asteroid: 100kg': {angle_difference:.5e} degrees.")


def plot_trajectories_with_ceres(fileNames, body_boundary_radius=None, ceres_radius=473000):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    max_distance = 0
    ceres_starting_position = None

    # Font size settings
    title_fontsize = 20
    label_fontsize = 16
    legend_fontsize = 14

    # Optionally draw the body boundary as a sphere (Earth)
    if body_boundary_radius is not None:
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        x = body_boundary_radius * np.cos(u) * np.sin(v)
        y = body_boundary_radius * np.sin(u) * np.sin(v)
        z = body_boundary_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="blue", label="Earth Boundary")

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            trajectories = simulation_data["trajectories"]

            # Update max_distance if necessary
            for path in trajectories.values():
                for position in path:
                    distance = np.linalg.norm(position)
                    if distance > max_distance:
                        max_distance = distance

            # Draw trajectories
            for name, path in trajectories.items():
                path = np.array(path)
                if name == "asteroid Ceres: 9.39×10^20":
                    ceres_starting_position = path[0]
                    ax.plot(path[::1000, 0], path[::1000, 1], path[::1000, 2], label="Asteroid Trajectories", color="red")

    # Check if the maximum trajectory distance is less than the provided body boundary radius
    if body_boundary_radius and body_boundary_radius > max_distance:
        max_distance = body_boundary_radius

    # Calculate axis scaling
    axisScaling = max_distance / AU

    # Set plot limits
    ax.set_xlim3d([-axisScaling * AU, axisScaling * AU])
    ax.set_ylim3d([-axisScaling * AU, axisScaling * AU])
    ax.set_zlim3d([-axisScaling * AU, axisScaling * AU])

    # Draw Ceres at its starting position to scale
    if ceres_starting_position is not None:
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = ceres_radius * np.cos(u) * np.sin(v) + ceres_starting_position[0]
        y = ceres_radius * np.sin(u) * np.sin(v) + ceres_starting_position[1]
        z = ceres_radius * np.cos(v) + ceres_starting_position[2]
        ax.plot_wireframe(x, y, z, color="green", label="Ceres")

    # Set aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set font sizes for labels, titles, and legends
    ax.set_title("Asteroid Trajectories", fontsize=title_fontsize)
    ax.set_xlabel('X-axis (m)', fontsize=label_fontsize)
    ax.set_ylabel('Y-axis (m)', fontsize=label_fontsize)
    ax.set_zlabel('Z-axis (m)', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize)

    plt.show()


def run_simulation(astroids):

    for astroid in astroids:
        print(f"Simulation for {astroid.name}")

        # Create solar system
        solar_system = SolarSystem()
        solar_system.add_body_at_center(CelestialBodyData.get_earth_data())
        solar_system.add_body_relative_to_other(astroid, "Earth", EARTH_RADIUS / 1000 * 3, 10, [30, 0])

        dt = 0.01  # Time step in seconds
        SIMULATION_TIME = 3600 * 3  # Seconds

        solar_system.run_simulation(timeStep=dt, SIMULATION_TIME=SIMULATION_TIME, integrator=solar_system.rk4_update, fileName=astroid.name)


def main_analyse_mass_impact():
    asteroids = [CelestialBodyData("Small asteroid: 100kg", "gray", 100, 0, 0),
                CelestialBodyData("Chicxulub crater: 10 billion tons", "red", 1e+13, 0, 0),
                CelestialBodyData("asteroid Bennu: 78 million tons", "yellow", 7.8e+10, 0, 0),
                CelestialBodyData("asteroid Ceres: 9.39×10^20", "green", 9.39e+20, 0, 0)]

    fileNames = [asteroid.name for asteroid in asteroids]

    run_simulation(asteroids)

    analyze_asteroid_trajectories(fileNames)

    plot_energy_and_angular_momentum(fileNames[0], step_interval=1000)

    plot_trajectories_with_ceres(fileNames=[astroid.name for astroid in asteroids], body_boundary_radius=EARTH_RADIUS)

    SolarSystem.animate_trajectories(SIMULATION_SPEED=2, fileName="Small asteroid: 100kg")


# Section for analysing the time step dependence
def simulate_time_step_sensitivity():
    # Define Ceres asteroid data
    ceres = CelestialBodyData("asteroid Ceres: 9.39×10^20", "green", 9.39e+20, 0, 0)
    integrator = SolarSystem.rk4_update  # Choose the integrator you want to analyze

    # Define a range of time steps for sensitivity analysis
    time_steps = [0.005, 0.01, 0.02, 0.05]  # Seconds
    SIMULATION_TIME = 3600 * 3  # Seconds, 3 hours simulated

    results = []

    for dt in time_steps:
        print(f"Running simulation for Ceres with time step {dt} using RK4 integrator")

        # Create a new solar system instance for each time step
        solar_system = SolarSystem()
        solar_system.add_body_at_center(CelestialBodyData.get_earth_data())
        solar_system.add_body_relative_to_other(ceres, "Earth", EARTH_RADIUS / 1000 * 3, 10, [30, 0])

        # Bind the solar_system instance to the integrator method
        bound_integrator = integrator.__get__(solar_system, SolarSystem)

        start_time = time.time()

        # Run the simulation
        solar_system.run_simulation(timeStep=dt, SIMULATION_TIME=SIMULATION_TIME, integrator=bound_integrator,
                                    fileName=f"Ceres_time_step_{dt}")

        # End timer and calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Simulation with time step {dt} completed in {elapsed_time:.2f} seconds")


def plot_time_step_dependence(fileNames, step_interval=10):
    # Initialize figures for plots
    fig_energy, ax_energy = plt.subplots(figsize=(15, 5))
    fig_angular, ax_angular = plt.subplots(figsize=(15, 5))

    # Variables to store final position differences
    final_position_differences = {}
    final_earth_movements = {}

    # Variables to store end positions for the 0.005-second time step
    reference_end_positions_ceres = None
    reference_end_positions_earth = None

    for fileName in fileNames:
        file_path = os.path.join(data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            # Extract initial positions
            initial_positions = np.array(simulation_data["trajectories"]["asteroid Ceres: 9.39×10^20"][0])
            initial_earth_position = np.array(simulation_data["trajectories"]["Earth"][0])

            # Energy Drift
            ke_history = np.array(simulation_data["kinetic_energy"])[::step_interval]
            pe_history = np.array(simulation_data["potential_energy"])[::step_interval]
            total_energy = ke_history + pe_history
            initial_total_energy = total_energy[0]
            energy_drift_percent = (total_energy - initial_total_energy) / initial_total_energy * 100
            ax_energy.plot(energy_drift_percent, label=f"{fileName} Energy Drift (%)")

            # Angular Momentum Z Drift
            angular_momentum = np.array(simulation_data["angular_momentum"])[::step_interval]
            initial_angular_momentum_z = angular_momentum[0, 2]  # Assuming Z is the 3rd component
            angular_momentum_z_drift = (angular_momentum[:, 2] - initial_angular_momentum_z) / initial_angular_momentum_z * 100
            ax_angular.plot(angular_momentum_z_drift, label=f"{fileName} Angular Momentum Z Drift (%)")

            # Store final position differences and earth movements
            final_positions = np.array(simulation_data["trajectories"]["asteroid Ceres: 9.39×10^20"][-1])
            final_earth_position = np.array(simulation_data["trajectories"]["Earth"][-1])
            final_position_differences[fileName] = np.linalg.norm(final_positions - initial_positions)
            final_earth_movements[fileName] = np.linalg.norm(final_earth_position - initial_earth_position)

            # Extract final positions
            final_position_ceres = np.array(simulation_data["trajectories"]["asteroid Ceres: 9.39×10^20"][-1])
            final_position_earth = np.array(simulation_data["trajectories"]["Earth"][-1])

            # Store final positions for 0.005-second timestep simulation
            if "0.005" in fileName:
                reference_final_position_ceres = final_position_ceres
                reference_final_position_earth = final_position_earth

    # Set plot properties for Energy Drift
    ax_energy.set_title("Total Energy Drift (%)", fontsize=20)
    ax_energy.set_xlabel("Step", fontsize=16)
    ax_energy.set_ylabel("Energy Drift (%)", fontsize=16)
    ax_energy.legend(fontsize=14)
    ax_energy.tick_params(axis='both', which='major', labelsize=14)

    # Set plot properties for Angular Momentum Z Drift
    ax_angular.set_title("Angular Momentum Z-axis Drift (%)", fontsize=20)
    ax_angular.set_xlabel("Step", fontsize=16)
    ax_angular.set_ylabel("Angular Momentum Drift (%)", fontsize=16)
    ax_angular.legend(fontsize=14)
    ax_angular.tick_params(axis='both', which='major', labelsize=14)

    plt.show()

    # Print the final position differences relative to the 0.005-second timestep simulation
    for fileName in fileNames:
        if "0.005" not in fileName:
            file_path = os.path.join(data_directory, f"{fileName}.json")
            with open(file_path, "r") as file:
                simulation_data = json.load(file)


                # Extract final positions
                final_position_ceres = np.array(simulation_data["trajectories"]["asteroid Ceres: 9.39×10^20"][-1])
                final_position_earth = np.array(simulation_data["trajectories"]["Earth"][-1])

                # Calculate the differences
                ceres_position_difference = np.linalg.norm(final_position_ceres - reference_final_position_ceres)
                earth_position_difference = np.linalg.norm(final_position_earth - reference_final_position_earth)

                print(f"{fileName}: Relative Ceres End Position Difference: {ceres_position_difference:.5f} meters")
                print(f"{fileName}: Relative Earth Movement: {earth_position_difference:.5f} meters")


def main_compare_time_steps():
    simulate_time_step_sensitivity()

    fileNames = ["Ceres_time_step_0.005", "Ceres_time_step_0.01", "Ceres_time_step_0.02", "Ceres_time_step_0.05"]  # Update with your actual file names
    plot_time_step_dependence(fileNames)


if __name__ == "__main__":
    main_analyse_mass_impact()
    main_compare_integrators()
    main_compare_time_steps()
