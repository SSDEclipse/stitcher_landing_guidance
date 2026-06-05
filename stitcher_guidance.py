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
        self.v_e = Isp * 9.80665


class Node:
    def __init__(self, position_array, velocity_array, time):
        self.position = position_array
        self.velocity = velocity_array
        self.time = time
        self.target_edges = []
        self.parent_edges = []
    
    # def create_and_append_edge(self, target_node):

    #     tf = target_node.arrival_time
    #     c0_array = 1.0/6.0 * tf**3 * (target_node.velocity - self.velocity) - 1.0/2.0 * tf**2 * (target_node.position - self.velocity*tf - self.position)
    #     c1_array = -1.0/2.0 * tf**2 * (target_node.velocity - self.velocity) + tf * (target_node.position - self.velocity*tf - self.position)
        
    #     new_edge = Edge(c0_array, c1_array, tf)
    #     self.edges.append(new_edge)
    #     return new_edge
    
    def create_and_append_constant_accel_edge(self, target_node, mode):
        if mode == 'forward':
            t_f = target_node.time
            if t_f == 0:
                c_0_array = np.array([0.0, 0.0, 0.0])
                v_f = self.velocity
                target_node.update_node_velocity(v_f)
            else:
                c_0_array = 2.0 / t_f**2 * (target_node.position - self.position - self.velocity * t_f)
                v_f = self.velocity + c_0_array*t_f
                c_0_array[0] += planetary_body_config.body_surface_gravity
                target_node.update_node_velocity(v_f)
            new_edge = Edge(self, target_node, c_0_array, np.array([0.0, 0.0, 0.0]), t_f)
            self.target_edges.append(new_edge)
            target_node.parent_edges.append(new_edge)
            return new_edge
        
        elif mode == 'backward':
            t_f = target_node.time
            t_f = target_node.time
            if t_f == 0:
                c_0_array = np.array([0.0, 0.0, 0.0])
                v_0 = target_node.velocity
                self.update_node_velocity(v_0)
            else:
                c_0_array = -2.0 / t_f**2 * (target_node.position - self.position - target_node.velocity * t_f)
                v_0 = target_node.velocity - c_0_array*t_f
                c_0_array[0] += planetary_body_config.body_surface_gravity
                self.update_node_velocity(v_0)
            new_edge = Edge(self, target_node, c_0_array, np.array([0.0, 0.0, 0.0]), t_f)
            self.target_edges.append(new_edge)
            target_node.parent_edges.append(new_edge)
            return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity



class Edge:
    def __init__(self, parent_node, target_node, c0_array, c1_array, t_f):
        self.parent_node = parent_node
        self.target_node = target_node
        self.c0_array = c0_array
        self.c1_array = c1_array
        self.t_f = t_f
        self.mass_consumed = None
        self.end_mass = None


    def compute_const_accel_mass_consumed(self, edge_start_mass, v_e):

        thrust_integral = edge_start_mass * v_e * (1 - np.exp(-np.linalg.norm(self.c0_array)* self.t_f / v_e))
        
        self.mass_consumed = thrust_integral / v_e

        self.end_mass = edge_start_mass - self.mass_consumed

    def check_const_accel_thrust_bounds(self, edge_start_mass, thrust_bound_lower, thrust_bound_upper):
        accel_mag = np.linalg.norm(self.c0_array)
        # only need to check thrust at start of edge since that's when it would
        # be highest for constant accel
        max_thrust_mag = accel_mag * edge_start_mass
        if max_thrust_mag > thrust_bound_upper or max_thrust_mag < thrust_bound_lower:
            return False
        return True

    # def check_linear_accel_thrust_bounds(self, thrust_bound_lower, thrust_bound_upper):
    #     t_array = np.linspace(0, self.t_f, 10)
    #     for time in t_array:
    #         accel_mag = np.linalg.norm(self.c0_array + self.c1_array * time)
    #         thrust_mag = accel_mag * mass
    #         if thrust_mag > thrust_bound_upper or thrust_mag < thrust_bound_lower:
    #             return False
    #     return True





def generate_position_set(pos_x, pos_y, pos_z):
    output = []
    for i in range(len(pos_x)):
        for j in range(len(pos_y)):
            for k in range(len(pos_z)):
                output.append(np.array([pos_x[i], pos_y[j], pos_z[j]]))
    output = np.asarray(output)
    return output




def generate_phase_1_nodes(initial_r):

    sampled_time_array = np.linspace(0.0, 10, 5)
    sampled_position_x_array = np.linspace(0, initial_r[0], 5)
    sampled_position_y_array = np.linspace(0, initial_r[1], 5)
    sampled_position_z_array = np.linspace(0, initial_r[2], 5)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_1_nodes = []
    for time in sampled_time_array:
        for pos in sampled_positions_array:
            phase_1_nodes.append(Node(pos, None, time))
    return phase_1_nodes

def generate_phase_2_nodes(initial_r):

    sampled_time_array = np.linspace(0.1, 10, 5)
    sampled_position_x_array = np.linspace(0, initial_r[0], 5)
    sampled_position_y_array = np.linspace(0, initial_r[1], 5)
    sampled_position_z_array = np.linspace(0, initial_r[2], 5)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_2_nodes = []
    for time in sampled_time_array:
        for pos in sampled_positions_array:
            phase_2_nodes.append(Node(pos, None, time))
    return phase_2_nodes

def generate_phase_3_nodes(target_r, target_v):

    sampled_time_array = np.linspace(0.1, 10, 5)

    phase_3_nodes = []
    for time in sampled_time_array:
        phase_3_nodes.append(Node(target_r, target_v, time))
    return phase_3_nodes


def prune_edges(input_mass_array, thrust_lower_bound, thrust_upper_bound, input_edges, input_nodes=None):

    # for edge in input_edges[:]:
    #     prune_edge_flag = False

    #     # check all constraints here
    #     if not edge.check_const_accel_thrust_bounds(input_mass, thrust_lower_bound, thrust_upper_bound):
    #         prune_edge_flag = True
    #     # elif ...

    #     if prune_edge_flag:
    #         input_edges.remove(edge)
    #         if not input_nodes == None:
    #             input_nodes.remove(edge.target_node)
    input_edges_copy = input_edges[:]
    for edg_idx in range(len(input_edges_copy)):
        edge = input_edges_copy[edg_idx]
        prune_edge_flag = False

        # check all constraints here
        if not edge.check_const_accel_thrust_bounds(input_mass_array[edg_idx], thrust_lower_bound, thrust_upper_bound):
            prune_edge_flag = True
        # elif ...

        if prune_edge_flag:
            input_edges.remove(edge)
            if not input_nodes == None:
                input_nodes.remove(edge.target_node)

def compute_edge_costs(v_e, input_edges):
    for edge in input_edges:
        edge.compute_const_accel_mass_consumed(v_e)





def generate_stitcher_trajectory_constant_accel(vehicle, initial_r, initial_v, final_r, final_v):

    start_node = Node(initial_r, initial_v, 0.0)

    # create phase 1 nodes

    phase_1_nodes = generate_phase_1_nodes(initial_r)

    # connect start and phase 1 nodes with edges
    for phase_1_node in phase_1_nodes:
        start_node.create_and_append_constant_accel_edge(phase_1_node, mode='forward')

    # compute costs of phase 1 edges
    for start_edge in start_node.target_edges:
        start_edge.compute_const_accel_mass_consumed(vehicle.wet_mass, vehicle.v_e)

    # prune phase 1 edges and nodes that violate constraints
    prune_edges([vehicle.wet_mass]*len(start_node.target_edges), vehicle.min_thrust, vehicle.max_thrust, start_node.target_edges, phase_1_nodes)
    
    # create phase 3 nodes
    phase_3_nodes = generate_phase_3_nodes(final_r, final_v)

    # create phase 2 nodes
    phase_2_nodes = generate_phase_2_nodes(initial_r)

    # connect phase 3 and phase 2 nodes with edges
    for phase_2_node in phase_2_nodes:
        for phase_3_node in phase_3_nodes:
            phase_2_node.create_and_append_constant_accel_edge(phase_3_node, mode='backward')

    # connect phase 1 and phase 2 nodes with edges
    for phase_1_node in phase_1_nodes:
        for phase_2_node in phase_2_nodes:
            phase_1_node.create_and_append_constant_accel_edge(phase_2_node, mode='forward')





    # compute costs of phase 2 edges
    counter = 0
    for phase_1_node in phase_1_nodes:
        for edg_idx in range(len(phase_1_node.target_edges)):
            phase_1_node.target_edges[edg_idx].compute_const_accel_mass_consumed(phase_1_node.parent_edges[0].end_mass, vehicle.v_e)
            counter += 1
    print(counter)

    # compute costs of phase 3 edges
    counter = 0
    for phase_2_node in phase_2_nodes:
        for edg_idx in range(len(phase_2_node.target_edges)):
            phase_2_node.target_edges[edg_idx].compute_const_accel_mass_consumed(phase_2_node.parent_edges[edg_idx].end_mass, vehicle.v_e)
            counter += 1
    print(counter)

    




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