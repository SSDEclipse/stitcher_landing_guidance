import numpy as np
import orbital_conversions
import orbital_dynamics
import cartesian_dynamics
import planetary_body_config


class Vehicle:
    def __init__(self, wet_mass, dry_mass, max_thrust, min_thrust, Isp):
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.Isp = Isp


class Node:
    def __init__(self, position_array, velocity_array, time, mass):
        self.position = position_array
        self.velocity = velocity_array
        self.time = time
        self.mass = mass
        self.edges = []
    
    # def create_and_append_edge(self, target_node):

    #     tf = target_node.arrival_time
    #     c0_array = 1.0/6.0 * tf**3 * (target_node.velocity - self.velocity) - 1.0/2.0 * tf**2 * (target_node.position - self.velocity*tf - self.position)
    #     c1_array = -1.0/2.0 * tf**2 * (target_node.velocity - self.velocity) + tf * (target_node.position - self.velocity*tf - self.position)
        
    #     new_edge = Edge(c0_array, c1_array, tf)
    #     self.edges.append(new_edge)
    #     return new_edge
    
    def create_and_append_constant_accel_edge(self, target_node):
        t_f = target_node.arrival_time
        c_0_array = 2.0 / t_f**2 * (target_node.position - self.position - self.velocity * t_f)
        c_0_array[0] += planetary_body_config.body_surface_gravity
        new_edge = Edge(target_node, c_0_array, np.array([0.0, 0.0, 0.0]), t_f)
        self.edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

    def update_node_mass(self, input_mass):
        self.mass = input_mass


class Edge:
    def __init__(self, target_node, c0_array, c1_array, t_f):
        self.target_node = target_node
        self.c0_array = c0_array
        self.c1_array = c1_array
        self.t_f = t_f

    def cost(self):
        accel_squared_integral = 0
        for i in range(len(self.c0_array)):
            accel_squared_integral += self.c0_array[i]**2 * self.t_f + self.c0_array[i]*self.c1_array[i] * self.t_f**2 + 1.0/3.0*self.c1_array[i]**2 * self.t_f**3
        # TODO: fix with thrust integral instead
        return accel_squared_integral

    
    def check_thrust_bounds(self, mass, thrust_bound_lower, thrust_bound_upper):
        t_array = np.linspace(0, self.t_f, 10)
        for time in t_array:
            accel_mag = self.c0_array + self.c1_array * time
            thrust_mag = accel_mag * mass
            if thrust_mag > thrust_bound_upper or thrust_mag < thrust_bound_lower:
                return False
        return True


def generate_discrete_set(lower_bound, upper_bound, num_points):
    return np.linspace(lower_bound, upper_bound, num_points)

def generate_time_set(discrete_time_array):
    return discrete_time_array

def generate_position_set(pos_x, pos_y, pos_z):
    output = []
    for i in range(len(pos_x)):
        for j in range(len(pos_y)):
            for k in range(len(pos_z)):
                output.append(np.array([pos_x[i], pos_y[j], pos_z[j]]))
    output = np.asarray(output)
    return output

def generate_velocity_set(vel_zenith, vel_azimuth, vel_mag):
    output = []
    for i in range(len(vel_zenith)):
        for j in range(len(vel_azimuth)):
            for k in range(len(vel_mag)):
                vel_x = vel_mag[k] * np.sin(vel_zenith[i]) * np.cos(vel_azimuth[j])
                vel_y = vel_mag[k] * np.sin(vel_zenith[i]) * np.sin(vel_azimuth[j])
                vel_z = vel_mag[k] * np.cos(vel_zenith[i])
                output.append(np.array([vel_x, vel_y, vel_z]))
    output = np.asarray(output)
    return output



def generate_phase_1_nodes(initial_r):

    sampled_time_array = generate_discrete_set(0.1, 10, 5)
    sampled_position_x_array = generate_discrete_set(0, initial_r[0], 5)
    sampled_position_y_array = generate_discrete_set(0, initial_r[1], 5)
    sampled_position_z_array = generate_discrete_set(0, initial_r[2], 5)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_1_nodes = []
    for time in sampled_time_array:
        for pos in sampled_positions_array:
            phase_1_nodes.append(Node(pos, None, time, None))


def generate_stitcher_trajectory_constant_accel(vehicle, initial_r, initial_v, target_r, target_v):

    start_node = Node(initial_r, initial_v, 0.0, vehicle.wet_mass)

    # phase 1

    generate_phase_1_nodes(initial_r)


    def generate_phase_1_edges_nodes(time_array, position_set, velocity_set):

        phase_2_nodes = []

        for time in time_array:
            for position in position_set:
                for velocity in velocity_set:
                    phase_2_nodes.append(Node(position, velocity, time))
        for node in phase_2_nodes:
            start_node.create_and_append_edge(node)
        return phase_2_nodes




# sampled_velocity_zenith_array = generate_discrete_set(np.pi/2, np.pi, 20)
# sampled_velocity_azimuth_array = generate_discrete_set(0, 2*np.pi, 30)
# sampled_velocity_magnitude_array = generate_discrete_set(0, 100, 20)
# sampled_position_x_array = generate_discrete_set(0, 100, 20)
# sampled_position_y_array = generate_discrete_set(0, 100, 20)
# sampled_position_z_array = generate_discrete_set(0, 100, 20)
# sampled_velocities_array = generate_velocity_set(sampled_velocity_zenith_array, sampled_velocity_azimuth_array, sampled_velocity_magnitude_array)
# sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

# print(np.shape(sampled_positions_array))
# print(1000*1000*20 / 1000000.0)
# for i in range(100000000):
#     test = 219586**22.1 + 230486*3409723 / 23047.987


lander = Vehicle(2000, 1000, 10000, 3000, 300)

generate_stitcher_trajectory_constant_accel(lander, np.array([100.0, 0.0, 0.0]), np.array([-10.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))