import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os


class CelestialBodyData:
    AU = np.float64(1.496e+11)

    def __init__(self, name, colour, mass, apoapsis, inclination):
        self.name = name
        self.colour = colour
        self.mass = np.float64(mass)
        self.apoapsis = np.float64(apoapsis)
        self.inclination = np.float64(inclination)

    @staticmethod
    def get_sun_data():
        return CelestialBodyData("Sun", "yellow", 1.9885e+30, 0.0,
                                 0.0)  # Sun's apoapsis and inclination are set to zero

    @staticmethod
    def get_mercury_data():
        return CelestialBodyData("Mercury", "gray", 3.3011e+23, 0.466 * CelestialBodyData.AU, 7.005)

    @staticmethod
    def get_venus_data():
        return CelestialBodyData("Venus", "yellow", 4.8675e+24, 0.728 * CelestialBodyData.AU, 3.39458)

    @staticmethod
    def get_earth_data():
        return CelestialBodyData("Earth", "blue", 5.9722e+24, 1.017 * CelestialBodyData.AU, 7.155)

    @staticmethod
    def get_mars_data():
        return CelestialBodyData("Mars", "red", 6.4171e+23, 1.666 * CelestialBodyData.AU, 1.85)

    @staticmethod
    def get_jupiter_data():
        return CelestialBodyData("Jupiter", "brown", 1.8982e+27, 5.455 * CelestialBodyData.AU, 1.303)

    @staticmethod
    def get_saturn_data():
        return CelestialBodyData("Saturn", "goldenrod", 5.6834e+26, 10.1238 * CelestialBodyData.AU, 2.485)

    @staticmethod
    def get_uranus_data():
        return CelestialBodyData("Uranus", "cyan", 8.6810e+25, 20.11 * CelestialBodyData.AU, 0.773)

    @staticmethod
    def get_neptune_data():
        return CelestialBodyData("Neptune", "blue", 1.02413e+26, 30.33 * CelestialBodyData.AU, 1.767)


class CelestialBody:
    def __init__(self, name, colour, mass, helio_position, helio_velocity, apoapsis=None, inclination=0):
        self.name = name
        self.colour = colour
        self.mass = np.float64(mass)
        self.apoapsis = np.float64(apoapsis) if apoapsis is not None else None
        self.inclination = np.radians(inclination)  # Convert inclination to radians, already in float64
        self.position = np.array(helio_position, dtype=np.float64)
        self.velocity = np.array(helio_velocity, dtype=np.float64)

    def set_initial_conditions(self, central_body):
        if self.apoapsis is not None:
            # Standard gravitational parameter
            mu = np.float64(SolarSystem.G * central_body.mass)

            central_position = central_body.position
            central_velocity = central_body.velocity

            # Inclined position at apoapsis
            self.position = central_position + np.array([self.apoapsis * np.cos(self.inclination),
                                                         0,
                                                         self.apoapsis * np.sin(self.inclination)], dtype=np.float64)

            # Velocity at apoapsis (simplified calculation)
            self.velocity = central_velocity + np.array([0,
                                                         np.sqrt(mu / self.apoapsis),
                                                         0], dtype=np.float64)


class SolarSystem:
    G = np.float64(6.67430e-11)  # Gravitational constant
    AU = np.float64(1.496e+11)  # Astronomical Unit in meters

    sun_mass = np.float64(1.9885e+30)  # Mass of the sun in kg

    data_directory = "SimulationData"

    def __init__(self):
        self.bodies = []
        self.ke_history = []
        self.pe_history = []
        self.angular_momentum_history = []

        # Initialize a dictionary to store trajectories
        self.trajectories = {}

    def get_body(self, name):
        for body in self.bodies:
            if body.name == name:
                return body

        raise NameError(f"A celestial body with the name '{name}' does not exist in the solar system.")

    def add_body(self, body):
        self.bodies.append(body)

    def add_body_at_center(self, body_data):
        body = CelestialBody(body_data.name, body_data.colour, body_data.mass, [0, 0, 0], [0, 0, 0])
        self.add_body(body)

    def add_body_in_orbit(self, body_data, central_body_name):
        central_body = self.get_body(central_body_name)
        body = CelestialBody(body_data.name, body_data.colour, body_data.mass, [0, 0, 0], [0, 0, 0],
                             body_data.apoapsis, body_data.inclination)
        body.set_initial_conditions(central_body)
        self.add_body(body)

    def add_body_relative_to_other(self, body_data, relative_body_name, distance_km, velocity_kms, attack_angle_deg):
        relative_body = self.get_body(relative_body_name)
        if relative_body is None:
            raise ValueError(f"Relative body '{relative_body_name}' not found.")

        body = CelestialBody(body_data.name, body_data.colour, body_data.mass, [0, 0, 0], [0, 0, 0],
                             body_data.apoapsis, body_data.inclination)

        # Convert distance from km to meters
        distance_m = distance_km * 1000

        # Set initial position of the new body
        body.position = relative_body.position + np.array([distance_m, 0, 0], dtype=np.float64)

        # Convert velocity from km/s to m/s and attack angles from degrees to radians
        velocity_ms = velocity_kms * 1000
        attack_angle_rad = np.radians(attack_angle_deg)

        # Initial velocity vector pointing towards the relative body
        velocity_vector = relative_body.position - body.position

        if attack_angle_deg != [0, 0]:
            # Adjust the velocity vector based on attack angles
            # Rotate in xy-plane (azimuth)
            rot_matrix_xy = np.array([
                [np.cos(attack_angle_rad[0]), -np.sin(attack_angle_rad[0]), 0],
                [np.sin(attack_angle_rad[0]), np.cos(attack_angle_rad[0]), 0],
                [0, 0, 1]
            ])

            # Rotate in xz-plane (elevation)
            rot_matrix_xz = np.array([
                [np.cos(attack_angle_rad[1]), 0, -np.sin(attack_angle_rad[1])],
                [0, 1, 0],
                [np.sin(attack_angle_rad[1]), 0, np.cos(attack_angle_rad[1])]
            ])

            # Apply rotations to the velocity vector
            velocity_vector = rot_matrix_xz @ rot_matrix_xy @ velocity_vector

        # Normalize and scale the velocity vector
        body.velocity = (velocity_vector / np.linalg.norm(velocity_vector)) * velocity_ms

        self.add_body(body)

    def calculate_energy_and_momentum(self):
        total_ke = 0
        total_pe = 0
        total_angular_momentum = np.zeros(3)
        counted_pairs = set()  # Keep track of pairs for which we've calculated potential energy

        for body in self.bodies:
            ke = 0.5 * body.mass * np.linalg.norm(body.velocity) ** 2
            total_ke += ke

            # Calculate the angular momentum with respect to the system's center of mass
            angular_momentum = np.cross(body.position, body.velocity) * body.mass
            total_angular_momentum += angular_momentum

            # Calculate potential energy between all unique pairs
            for other_body in self.bodies:
                if other_body != body and (other_body, body) not in counted_pairs:
                    pe = -self.G * body.mass * other_body.mass / np.linalg.norm(body.position - other_body.position)
                    total_pe += pe
                    counted_pairs.add((body, other_body))  # Mark this pair as counted

        self.ke_history.append(total_ke)
        self.pe_history.append(total_pe)
        self.angular_momentum_history.append(total_angular_momentum)

    def euler_update(self, dt):
        for body in self.bodies:
            acceleration = self._compute_acceleration(body, body.position)
            body.velocity += acceleration * dt
            body.position += body.velocity * dt

    def verlet_update(self, dt):
        for body in self.bodies:
            if not hasattr(body, 'prev_position'):
                body.prev_position = body.position - body.velocity * dt  # Approximate if not set

            acceleration = self._compute_acceleration(body, body.position)
            new_position = 2 * body.position - body.prev_position + acceleration * dt ** 2
            new_velocity = (new_position - body.prev_position) / (2 * dt)  # Corrected velocity calculation

            body.prev_position = body.position
            body.position = new_position
            body.velocity = new_velocity  # Update the velocity using the corrected calculation

    def rk4_update(self, dt):
        initial_states = [(body.position, body.velocity) for body in self.bodies]

        k1_velocities, k1_positions = self._compute_k1(dt, initial_states)
        k2_velocities, k2_positions = self._compute_k(dt, initial_states, k1_velocities, 0.5 * dt)
        k3_velocities, k3_positions = self._compute_k(dt, initial_states, k2_velocities, 0.5 * dt)
        k4_velocities, k4_positions = self._compute_k(dt, initial_states, k3_velocities, dt)

        for i, body in enumerate(self.bodies):
            position_update = (k1_positions[i] + 2 * k2_positions[i] + 2 * k3_positions[i] + k4_positions[i]) / 6
            velocity_update = (k1_velocities[i] + 2 * k2_velocities[i] + 2 * k3_velocities[i] + k4_velocities[i]) / 6

            body.position += position_update
            body.velocity += velocity_update

    def _compute_k1(self, dt, initial_states):
        k1_velocities = []
        k1_positions = []
        for body, (position, velocity) in zip(self.bodies, initial_states):
            acceleration = self._compute_acceleration(body, position)
            k1_velocities.append(acceleration * dt)
            k1_positions.append(velocity * dt)
        return k1_velocities, k1_positions

    def _compute_k(self, dt, initial_states, k_velocities, dt_factor):
        k_positions = []
        k_velocities_updated = []
        for i, (body, (position, velocity)) in enumerate(zip(self.bodies, initial_states)):
            new_velocity = velocity + k_velocities[i] * dt_factor
            new_position = position + new_velocity * dt
            acceleration = self._compute_acceleration(body, new_position)
            k_velocities_updated.append(acceleration * dt)
            k_positions.append(new_velocity * dt)
        return k_velocities_updated, k_positions

    def _compute_acceleration(self, body, position):
        acceleration = np.zeros(3)
        for other_body in self.bodies:
            if other_body != body:
                force = self.compute_gravitational_force(body, other_body, position)
                acceleration += force / body.mass
        return acceleration

    def compute_gravitational_force(self, body1, body2, position):
        r = body2.position - position
        distance = np.linalg.norm(r)
        # softening_length = 1e-6  # Prevents division by zero
        # force = self.G * body1.mass * body2.mass / (distance**2 + softening_length**2)
        force = self.G * body1.mass * body2.mass / distance ** 2
        return force * r / distance

    def get_current_positions(self):
        return {body.name: body.position for body in self.bodies}

    def get_current_velocity(self):
        return {body.name: body.velocity for body in self.bodies}

    def run_simulation(self, timeStep, SIMULATION_TIME, integrator, fileName="Trajectories"):
        self.trajectories = {body.name: [] for body in self.bodies}
        self.velocities = {body.name: [] for body in self.bodies}

        days_per_frame = timeStep / (24 * 3600)  # Days per frame

        SIMULATION_STEPS = int(SIMULATION_TIME / timeStep)
        print(f"Number of steps {SIMULATION_STEPS}")
        for i in range(SIMULATION_STEPS):
            integrator(timeStep)  # Call the passed integrator function

            self.calculate_energy_and_momentum()

            positions = self.get_current_positions()
            velocities = self.get_current_velocity()

            for name in self.trajectories:
                self.trajectories[name].append(positions[name].copy())
                self.velocities[name].append(velocities[name].copy())

            if i % round(SIMULATION_STEPS / 10) == 0:
                print(f"{round(i / SIMULATION_STEPS * 100)} % done")

        print("100% done")

        # Store additional simulation data
        simulation_data = {"trajectories": self.trajectories,
                           "velocities": self.velocities,
                           "kinetic_energy": [np.array(ke).tolist() for ke in self.ke_history],
                           "potential_energy": [np.array(pe).tolist() for pe in self.pe_history],
                           "angular_momentum": [np.array(am).tolist() for am in self.angular_momentum_history],
                           "bodies": [{"name": body.name, "colour": body.colour} for body in self.bodies],
                           "steps": SIMULATION_STEPS, "days_per_frame": days_per_frame, "animation_interval": 1}

        # Convert numpy arrays to lists for JSON serialization
        for name in simulation_data["trajectories"]:
            simulation_data["trajectories"][name] = [pos.tolist() for pos in simulation_data["trajectories"][name]]

        for name in simulation_data["velocities"]:
            simulation_data["velocities"][name] = [vel.tolist() for vel in simulation_data["velocities"][name]]

        # Create 'SimulationData' directory if it does not exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # Save the simulation data to the JSON file within 'SimulationData' directory
        file_path = os.path.join(self.data_directory, f"{fileName}.json")
        with open(file_path, "w") as file:
            json.dump(simulation_data, file)

        print(f"Simulation data saved to '{file_path}'")

    def plot_energy_and_angular_momentum(self, fileName="Trajectories"):
        file_path = os.path.join(self.data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            ke_history = simulation_data["kinetic_energy"]
            pe_history = simulation_data["potential_energy"]
            angular_momentum = simulation_data["angular_momentum"]

            # Calculate the total energy (kinetic + potential) at each step
            total_energy = [ke + pe for ke, pe in zip(ke_history, pe_history)]

            # Calculate the initial total energy
            initial_total_energy = total_energy[0]

            # Calculate the energy drift as a percentage of the initial total energy
            energy_drift_percent = [(e - initial_total_energy) / initial_total_energy * 100 for e in total_energy]

            print(
                f"Total Energy Drift {energy_drift_percent[0] - energy_drift_percent[len(energy_drift_percent) - 1]} %")
            print(f"Total Energy {initial_total_energy} Joules")

            # Create a figure with 2 subplots (energy drift in percent and angular momentum)
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            # Energy Drift in Percent Plot
            axs[0].plot(energy_drift_percent, label="Energy Drift (%)")
            axs[0].set_title("Energy Drift (%)")
            axs[0].set_xlabel("Step")
            axs[0].set_ylabel("Energy Drift (%)")

            # Angular Momentum Plot
            axs[1].plot(angular_momentum, label="Angular Momentum")
            axs[1].set_title("Angular Momentum")
            axs[1].set_xlabel("Step")
            axs[1].set_ylabel("Momentum")

            # Show the legend and plot
            for ax in axs:
                ax.legend()
            plt.show()

    def plot_trajectories(self, fileName="Trajectories", body_boundary_radius=None):
        file_path = os.path.join(self.data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            trajectories = simulation_data["trajectories"]

            # Find the maximum distance any Body reaches from the origin
            max_distance = 0
            for path in trajectories.values():
                for position in path:
                    distance = np.linalg.norm(position)
                    if distance > max_distance:
                        max_distance = distance

            if body_boundary_radius > max_distance:
                max_distance = body_boundary_radius

            # Calculate axis scaling
            axisScaling = max_distance / SolarSystem.AU

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-axisScaling * SolarSystem.AU, axisScaling * SolarSystem.AU])
            ax.set_ylim3d([-axisScaling * SolarSystem.AU, axisScaling * SolarSystem.AU])
            ax.set_zlim3d([-axisScaling * SolarSystem.AU, axisScaling * SolarSystem.AU])

            # Set aspect ratio to be equal
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

            # Draw trajectories
            for name, path in trajectories.items():
                path = np.array(path)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], label=name)

            # Optionally draw the body boundary as a sphere
            if body_boundary_radius is not None:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = body_boundary_radius * np.cos(u) * np.sin(v)
                y = body_boundary_radius * np.sin(u) * np.sin(v)
                z = body_boundary_radius * np.cos(v)
                ax.plot_wireframe(x, y, z, color="blue")

            ax.legend()
            plt.show()

    def update_frame(self, num, graph, trajectories, days_text, days_per_frame, frame_skip):
        step_index = num * frame_skip
        for name in graph.keys():
            graph[name].set_data(trajectories[name][step_index][0], trajectories[name][step_index][1])
            graph[name].set_3d_properties(trajectories[name][step_index][2])
        current_day = step_index * days_per_frame
        days_text.set_text(f"Day: {round(current_day, 1)}")
        return graph.values()

    def animate_trajectories(self, SIMULATION_SPEED=1, fileName="Trajectories"):

        SIMULATION_SPEED = round(SIMULATION_SPEED * 100)

        file_path = os.path.join(self.data_directory, f"{fileName}.json")

        with open(file_path, "r") as file:
            simulation_data = json.load(file)

            trajectories = simulation_data["trajectories"]
            bodies = simulation_data["bodies"]
            steps = simulation_data["steps"]
            days_per_frame = simulation_data["days_per_frame"]
            animation_interval = simulation_data["animation_interval"]

            # Find the maximum distance any Body reaches from the origin
            max_distance = 0
            for path in trajectories.values():
                for position in path:
                    distance = np.linalg.norm(position)
                    if distance > max_distance:
                        max_distance = distance

            # Calculate axis scaling
            axisScaling = max_distance / SolarSystem.AU

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-axisScaling * SolarSystem.AU, axisScaling * SolarSystem.AU])
            ax.set_ylim3d([-axisScaling * SolarSystem.AU, axisScaling * SolarSystem.AU])
            ax.set_zlim3d([-axisScaling * SolarSystem.AU, axisScaling * SolarSystem.AU])

            graph = {body['name']: ax.plot([], [], [], 'o', color=body['colour'], label=body['name'])[0]
                     for body in bodies}

            # Add a text element for the current day
            days_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

            original_animation_interval = animation_interval
            animation_interval = max(1, original_animation_interval // SIMULATION_SPEED)

            # Calculate total number of frames considering frame skip
            total_frames = int(steps / SIMULATION_SPEED)

            ani = animation.FuncAnimation(fig, self.update_frame,
                                          fargs=(graph, trajectories, days_text, days_per_frame, SIMULATION_SPEED),
                                          frames=total_frames, interval=animation_interval)
            plt.legend()
            plt.show()
