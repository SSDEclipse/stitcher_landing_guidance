import numpy as np
import orbital_conversions
import orbital_dynamics
import cartesian_dynamics
import planetary_body_config
import matplotlib.pyplot as plt
import plotting_functions
import copy


class Vehicle:
    def __init__(self, wet_mass, dry_mass, max_thrust, min_thrust, Isp):
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.Isp = Isp
        self.v_e = Isp * 9.80665


class StartNode:
    def __init__(self, position, velocity, acceleration, time):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.time = time
        self.target_edges = []
        self.parent_edges = []
    
    
    def create_and_append_edge(self, target_node):
        t_f = target_node.time
        if t_f == 0:
            c_0_array = np.array([0.0, 0.0, 0.0])
            c_1_array = np.array([0.0, 0.0, 0.0])
            v_f = self.velocity
            target_node.update_node_velocity(v_f)
        else:
            c_0_array = self.acceleration.copy()
            c_1_array = 6.0 / t_f**3 * (target_node.position - self.position - self.velocity * t_f - 0.5*t_f**2 * c_0_array)
            v_f = self.velocity + c_0_array * t_f + 0.5*t_f**2 * c_1_array
            c_0_array[0] += planetary_body_config.body_surface_gravity
            target_node.update_node_velocity(v_f)
        new_edge = Edge(self, target_node, c_0_array, c_1_array, t_f)
        self.target_edges.append(new_edge)
        target_node.parent_edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

class P1Node:
    def __init__(self, position, velocity, time):
        self.position = position
        self.velocity = velocity
        self.time = time
        self.target_edges = []
        self.parent_edges = []
    
    
    def create_and_append_edge(self, target_node):
        t_f = target_node.time
        if t_f == 0:
            c_0_array = np.array([0.0, 0.0, 0.0])
            c_1_array = np.array([0.0, 0.0, 0.0])
        else:
            c_0_array = 6.0 / t_f**2 * (target_node.position - self.position - self.velocity*t_f) - 2.0 / t_f * (target_node.velocity - self.velocity)
            c_1_array = -12.0 / t_f**3 * (target_node.position - self.position - self.velocity*t_f) + 6.0 / t_f**2 * (target_node.velocity - self.velocity)
            c_0_array[0] += planetary_body_config.body_surface_gravity
        new_edge = Edge(self, target_node, c_0_array, c_1_array, t_f)
        self.target_edges.append(new_edge)
        target_node.parent_edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

class P2Node:
    def __init__(self, position, velocity, time, time_to_p3):
        self.position = position
        self.velocity = velocity
        self.time = time
        self.time_to_p3 = time_to_p3
        self.target_edges = []
        self.parent_edges = []
    
    
    # def create_and_append_forward_edge(self, target_node):
    #     t_f = self.time_to_p3
    #     if t_f == 0:
    #         c_0_array = np.array([0.0, 0.0, 0.0])
    #         c_1_array = np.array([0.0, 0.0, 0.0])
    #         v_f = self.velocity
    #         target_node.update_node_velocity(v_f)
    #     else:
    #         c_0_array = self.acceleration
    #         c_1_x = 6 / t_f**3 * (target_node.position[0] - self.position[0] - self.velocity[0] * t_f + 0.5*t_f**2 * planetary_body_config.body_surface_gravity - 0.5*t_f**2 * c_0_array[0])
    #         c_1_y = 6 / t_f**3 * (target_node.position[1] - self.position[1] - self.velocity[1] * t_f - 0.5*t_f**2 * c_0_array[1])
    #         c_1_z = 6 / t_f**3 * (target_node.position[2] - self.position[2] - self.velocity[2] * t_f - 0.5*t_f**2 * c_0_array[2])
    #         v_f_x = self.velocity[0] + (c_0_array[0] - planetary_body_config.body_surface_gravity) * t_f + 0.5*t_f**2 * c_1_x
    #         v_f_y = self.velocity[1] + c_0_array[1] * t_f + 0.5*t_f**2 * c_1_y
    #         v_f_z = self.velocity[2] + c_0_array[2] * t_f + 0.5*t_f**2 * c_1_z
    #         c_1_array = np.array([c_1_x, c_1_y, c_1_z])
    #         v_f = np.array([v_f_x, v_f_y, v_f_z])
    #         target_node.update_node_velocity(v_f)
    #     new_edge = Edge(self, target_node, c_0_array, c_1_array, t_f)
    #     self.target_edges.append(new_edge)
    #     target_node.parent_edges.append(new_edge)
    #     return new_edge
    
    def create_and_append_backward_edge(self, target_node):
        t_f = self.time_to_p3
        if t_f == 0:
            c_0_array = np.array([0.0, 0.0, 0.0])
            c_1_array = np.array([0.0, 0.0, 0.0])
            v_0 = target_node.velocity
            self.update_node_velocity(v_0)
        else:
            c_0_array = -2.0 * target_node.acceleration - 6.0 / t_f**2 * (target_node.position - self.position - target_node.velocity*t_f)
            c_1_array = 3.0 / t_f * target_node.acceleration + 6 / t_f**3 * (target_node.position - self.position - target_node.velocity*t_f)
            v_0 = target_node.velocity - c_0_array*t_f - 0.5*t_f**2 * c_1_array
            c_0_array[0] += planetary_body_config.body_surface_gravity
            self.update_node_velocity(v_0)

        new_edge = Edge(self, target_node, c_0_array, c_1_array, t_f)
        self.target_edges.append(new_edge)
        target_node.parent_edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

class P3Node:
    def __init__(self, position, velocity, acceleration):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.target_edges = []
        self.parent_edges = []




class Edge:
    def __init__(self, parent_node, target_node, c_0_array, c_1_array, t_f):
        self.parent_node = parent_node
        self.target_node = target_node
        self.c_0_array = c_0_array
        self.c_1_array = c_1_array
        self.t_f = t_f
        self.mass_consumed = None
        self.end_mass = None


    def compute_const_accel_mass_consumed(self, edge_start_mass, v_e):

        thrust_integral = edge_start_mass * v_e * (1 - np.exp(-np.linalg.norm(self.c_0_array)* self.t_f / v_e))
        
        self.mass_consumed = thrust_integral / v_e

        self.end_mass = edge_start_mass - self.mass_consumed

    def compute_linear_accel_mass_consumed(self, edge_start_mass, v_e):

        start_accel_mag = np.linalg.norm(self.c_0_array)
        end_accel_mag = np.linalg.norm(self.c_0_array + self.c_1_array*self.t_f)
        avg_accel_mag = 0.5*(start_accel_mag + end_accel_mag)
        temp_thrust_integral = edge_start_mass * v_e * (1 - np.exp(-avg_accel_mag* self.t_f / v_e))
        temp_mass_consumed = temp_thrust_integral / v_e
        start_thrust_mag = start_accel_mag * edge_start_mass
        end_thrust_mag = end_accel_mag * (edge_start_mass - temp_mass_consumed)
        estimated_thrust_integral = 0.5 * (start_thrust_mag + end_thrust_mag) * self.t_f
        
        self.mass_consumed = estimated_thrust_integral / v_e

        self.end_mass = edge_start_mass - self.mass_consumed

    def check_const_accel_thrust_bounds(self, edge_start_mass, thrust_bound_lower, thrust_bound_upper):
        accel_mag = np.linalg.norm(self.c_0_array)
        # only need to check thrust at start of edge since that's when it would
        # be highest for constant accel
        max_thrust_mag = accel_mag * edge_start_mass
        if max_thrust_mag > thrust_bound_upper or max_thrust_mag < thrust_bound_lower:
            return False
        return True

    def check_linear_accel_thrust_bounds(self, edge_start_mass, v_e, thrust_bound_lower, thrust_bound_upper):
        # only checks start and end, but should probably also check a few points in between
        start_accel_mag = np.linalg.norm(self.c_0_array)
        end_accel_mag = np.linalg.norm(self.c_0_array + self.c_1_array*self.t_f)
        avg_accel_mag = 0.5*(start_accel_mag + end_accel_mag)
        temp_thrust_integral = edge_start_mass * v_e * (1 - np.exp(-avg_accel_mag* self.t_f / v_e))
        temp_mass_consumed = temp_thrust_integral / v_e
        start_thrust_mag = start_accel_mag * edge_start_mass
        end_thrust_mag = end_accel_mag * (edge_start_mass - temp_mass_consumed)
        print(self.c_0_array[0], (self.c_0_array + self.c_1_array*self.t_f)[0], self.parent_node.position[0], self.parent_node.velocity[0], self.target_node.position[0], self.target_node.velocity[0], self.t_f)
        if start_thrust_mag > thrust_bound_upper or start_thrust_mag < thrust_bound_lower or end_thrust_mag > thrust_bound_upper or end_thrust_mag < thrust_bound_lower:
            return False
        return True


class SampledSet:
    def __init__(self, lower_bound, upper_bound, num_points):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_points = num_points

class OutputData:
    def __init__(self, start_node, optimal_node_1, optimal_node_2, optimal_node_3, optimal_edge_1, optimal_edge_2, optimal_edge_3, p1_nearest_neighbors_dict, p2_nearest_neighbors_dict, total_edges, total_valid_edges, end_masses):
        self.start_node = start_node
        self.optimal_node_1 = optimal_node_1
        self.optimal_node_2 = optimal_node_2
        self.optimal_node_3 = optimal_node_3
        self.optimal_edge_1 = optimal_edge_1
        self.optimal_edge_2 = optimal_edge_2
        self.optimal_edge_3 = optimal_edge_3
        self.p1_nearest_neighbors_dict = p1_nearest_neighbors_dict
        self.p2_nearest_neighbors_dict = p2_nearest_neighbors_dict
        self.total_edges = total_edges
        self.total_valid_edges = total_valid_edges
        self.end_masses = end_masses


def get_nearest_array_neighbors(input_array, input_value):
    idx = np.searchsorted(input_array, input_value)
    
    if idx < len(input_array) and input_array[idx] == input_value:
        exact_idx = idx
    else:
        if idx == 0:
            exact_idx = 0
        elif idx == len(input_array):
            exact_idx = len(input_array) - 1
        else:
            exact_idx = idx if abs(input_array[idx] - input_value) < abs(input_array[idx-1] - input_value) else idx - 1

    start_idx = max(0, exact_idx - 1)
    end_idx = min(len(input_array)-1, exact_idx + 1) 
    
    return input_array[start_idx], input_array[end_idx]

def generate_position_set(pos_x, pos_y, pos_z):
    output = []
    for i in range(len(pos_x)):
        for j in range(len(pos_y)):
            for k in range(len(pos_z)):
                output.append(np.array([pos_x[i], pos_y[j], pos_z[k]]))
    output = np.asarray(output)
    return output

def generate_phase_1_nodes(time_sampled_set, pos_x_sampled_set, pos_y_sampled_set, pos_z_sampled_set):

    sampled_time_array = np.linspace(time_sampled_set.lower_bound, time_sampled_set.upper_bound, time_sampled_set.num_points)
    sampled_position_x_array = np.linspace(pos_x_sampled_set.lower_bound, pos_x_sampled_set.upper_bound, pos_x_sampled_set.num_points)
    sampled_position_y_array = np.linspace(pos_y_sampled_set.lower_bound, pos_y_sampled_set.upper_bound, pos_y_sampled_set.num_points)
    sampled_position_z_array = np.linspace(pos_z_sampled_set.lower_bound, pos_z_sampled_set.upper_bound, pos_z_sampled_set.num_points)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_1_nodes = []
    for time in sampled_time_array:
        for pos in sampled_positions_array:
            phase_1_nodes.append(P1Node(pos, None, time))
    return phase_1_nodes

def generate_phase_2_nodes(time_sampled_set, time_to_p3_sampled_set, pos_x_sampled_set, pos_y_sampled_set, pos_z_sampled_set):

    sampled_time_array = np.linspace(time_sampled_set.lower_bound, time_sampled_set.upper_bound, time_sampled_set.num_points)
    sampled_time_to_p3_array = np.linspace(time_to_p3_sampled_set.lower_bound, time_to_p3_sampled_set.upper_bound, time_to_p3_sampled_set.num_points)
    sampled_position_x_array = np.linspace(pos_x_sampled_set.lower_bound, pos_x_sampled_set.upper_bound, pos_x_sampled_set.num_points)
    sampled_position_y_array = np.linspace(pos_y_sampled_set.lower_bound, pos_y_sampled_set.upper_bound, pos_y_sampled_set.num_points)
    sampled_position_z_array = np.linspace(pos_z_sampled_set.lower_bound, pos_z_sampled_set.upper_bound, pos_z_sampled_set.num_points)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_2_nodes = []
    for time in sampled_time_array:
        for time_to_p3 in sampled_time_to_p3_array:
            for pos in sampled_positions_array:
                phase_2_nodes.append(P2Node(pos, None, time, time_to_p3))
    return phase_2_nodes

def generate_phase_3_nodes(target_r, target_v, final_a):


    phase_3_nodes = [P3Node(target_r, target_v, final_a)]
    
    return phase_3_nodes


# def prune_edge(edge, input_mass, thrust_lower_bound, thrust_upper_bound):
#         # check all constraints here
#         if not edge.check_const_accel_thrust_bounds(input_mass, thrust_lower_bound, thrust_upper_bound):
#             prune_edge_flag = True
#         # elif other constraints...

#         if prune_edge_flag:
#             input_edges.remove(edge)
#             if mode == 'forward':
#                 edge.target_node.parent_edges.remove(edge)
#                 if len(edge.target_node.parent_edges) == 0:
#                     input_nodes.remove(edge.target_node)
#             elif mode == 'backward':
#                 edge.parent_node.target_edges.remove(edge)
#                 if len(edge.parent_node.target_edges) == 0:
#                     input_nodes.remove(edge.parent_node)  

def prune_edges(input_mass_array, input_v_e, thrust_lower_bound, thrust_upper_bound, input_edges, mode, input_nodes=None):

    input_edges_copy = input_edges[:]
    for edg_idx in range(len(input_edges_copy)):
        edge = input_edges_copy[edg_idx]
        prune_edge_flag = False

        # check all constraints here
        if not edge.check_linear_accel_thrust_bounds(input_mass_array[edg_idx], input_v_e, thrust_lower_bound, thrust_upper_bound):
            prune_edge_flag = True
        # elif other constraints...

        if prune_edge_flag:
            input_edges.remove(edge)
            if mode == 'forward':
                edge.target_node.parent_edges.remove(edge)
                if len(edge.target_node.parent_edges) == 0:
                    input_nodes.remove(edge.target_node)
            elif mode == 'backward':
                edge.parent_node.target_edges.remove(edge)
                if len(edge.parent_node.target_edges) == 0:
                    input_nodes.remove(edge.parent_node)


def compute_edge_costs(v_e, input_edges):
    for edge in input_edges:
        edge.compute_linear_accel_mass_consumed(v_e)



def generate_stitcher_trajectory_constant_accel(vehicle, initial_r, initial_v, initial_a, final_r, final_v, final_a, p1_sampled_set_dict, p2_sampled_set_dict):

    start_node = StartNode(initial_r, initial_v, initial_a, 0.0)

    # create phase 1 nodes

    phase_1_nodes = generate_phase_1_nodes(p1_sampled_set_dict['time'], p1_sampled_set_dict['pos_x'], p1_sampled_set_dict['pos_y'], p1_sampled_set_dict['pos_z'])

    # connect start and phase 1 nodes with edges
    for phase_1_node in phase_1_nodes:
        start_node.create_and_append_edge(phase_1_node)

    # compute costs of phase 1 edges
    for start_edge in start_node.target_edges:
        start_edge.compute_linear_accel_mass_consumed(vehicle.wet_mass, vehicle.v_e)

    total_p1_edges = 0
    for phase_1_edge in start_node.target_edges:
        total_p1_edges += 1
    print(total_p1_edges)
    # breakpoint()
    # prune phase 1 edges and nodes that violate constraints
    prune_edges([vehicle.wet_mass]*len(start_node.target_edges), vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust, start_node.target_edges, 'forward', phase_1_nodes)
    total_p1_edges = 0
    for phase_1_edge in start_node.target_edges:
        total_p1_edges += 1
    print(total_p1_edges)
    # breakpoint()

    # create phase 3 nodes
    phase_3_nodes = generate_phase_3_nodes(final_r, final_v, final_a)

    # create phase 2 nodes
    phase_2_nodes = generate_phase_2_nodes(p2_sampled_set_dict['time'], p2_sampled_set_dict['time_to_p3'], p2_sampled_set_dict['pos_x'], p2_sampled_set_dict['pos_y'], p2_sampled_set_dict['pos_z'])


    # connect phase 3 and phase 2 nodes with edges
    for phase_2_node in phase_2_nodes:
        for phase_3_node in phase_3_nodes:
            phase_2_node.create_and_append_backward_edge(phase_3_node)
            # print(phase_2_node.velocity, phase_3_node.time)

    total_p3_edges = 0
    for phase_3_edge in phase_3_nodes[0].parent_edges:
        total_p3_edges += 1
    print(total_p3_edges)
    # breakpoint()
    # prune phase 3 edges that violate constraints based on worst case mass
    for phase_3_node in phase_3_nodes:
        prune_edges([vehicle.dry_mass]*len(phase_3_node.parent_edges), vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust, phase_3_node.parent_edges, 'backward', phase_2_nodes)
    total_p3_edges = 0
    for phase_3_edge in phase_3_nodes[0].parent_edges:
        total_p3_edges += 1
    print(total_p3_edges)
    # breakpoint()
    # connect phase 1 and phase 2 nodes with edges
    for phase_1_node in phase_1_nodes:
        for phase_2_node in phase_2_nodes:
            phase_1_node.create_and_append_edge(phase_2_node)


    # compute costs of phase 2 edges
    for phase_1_node in phase_1_nodes:
        for edg_idx in range(len(phase_1_node.target_edges)):
            phase_1_node.target_edges[edg_idx].compute_linear_accel_mass_consumed(phase_1_node.parent_edges[0].end_mass, vehicle.v_e)

    total_p2_edges = 0
    for phase_1_node in phase_1_nodes:
        for phase_2_edge in phase_1_node.target_edges:
            total_p2_edges += 1
    print(total_p2_edges)
    # breakpoint()
    # prune phase 2 edges that violate constraints
    for phase_1_node in phase_1_nodes:
        prune_edges([phase_1_node.parent_edges[0].end_mass]*len(phase_1_node.target_edges), vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust, phase_1_node.target_edges, 'forward', phase_2_nodes)
    total_p2_edges = 0
    for phase_1_node in phase_1_nodes:
        for phase_2_edge in phase_1_node.target_edges:
            total_p2_edges += 1
    print(total_p2_edges)
    # breakpoint()

    for phase_2_node in phase_2_nodes:
        if len(phase_2_node.target_edges) != 1:
            print(len(phase_2_node.target_edges))
            raise ValueError




    total_edges = 0
    for phase_1_node in phase_1_nodes:
        for phase_2_edge in phase_1_node.target_edges:
            total_edges += 1


    end_masses = []
    best_mass = 0.0
    total_valid_edges = 0
    for phase_2_node in phase_2_nodes:
        for parent_edge in phase_2_node.parent_edges:
            current_edge_start_mass = parent_edge.end_mass
            for target_edge in phase_2_node.target_edges: # should only be 1 p2 target edge
                # breakpoint()
                if target_edge.check_linear_accel_thrust_bounds(current_edge_start_mass, vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust):
                    target_edge.compute_linear_accel_mass_consumed(current_edge_start_mass, vehicle.v_e)
                    total_valid_edges += 1
                    touchdown_mass = target_edge.end_mass
                    end_masses.append(touchdown_mass)
                    if touchdown_mass > best_mass:
                        best_mass = touchdown_mass
                        optimal_node_3 = phase_3_nodes[0]
                        optimal_node_2 = copy.copy(phase_2_node)
                        optimal_node_1 = copy.copy(parent_edge.parent_node)
                        optimal_edge_3 = copy.copy(target_edge)
                        optimal_edge_2 = copy.copy(parent_edge)
                        optimal_edge_1 = copy.copy(optimal_node_1.parent_edges[0])
    print(total_valid_edges)
    print(best_mass)
    # breakpoint()
    
    new_p1_time_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p1_sampled_set_dict['time'].lower_bound, p1_sampled_set_dict['time'].upper_bound, p1_sampled_set_dict['time'].num_points), optimal_edge_1.t_f), 5)
    new_p1_pos_x_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p1_sampled_set_dict['pos_x'].lower_bound, p1_sampled_set_dict['pos_x'].upper_bound, p1_sampled_set_dict['pos_x'].num_points), optimal_node_1.position[0]), 5)
    new_p1_pos_y_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p1_sampled_set_dict['pos_y'].lower_bound, p1_sampled_set_dict['pos_y'].upper_bound, p1_sampled_set_dict['pos_y'].num_points), optimal_node_1.position[1]), 5)
    new_p1_pos_z_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p1_sampled_set_dict['pos_z'].lower_bound, p1_sampled_set_dict['pos_z'].upper_bound, p1_sampled_set_dict['pos_z'].num_points), optimal_node_1.position[2]), 5)

    new_p2_time_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p2_sampled_set_dict['time'].lower_bound, p2_sampled_set_dict['time'].upper_bound, p2_sampled_set_dict['time'].num_points), optimal_edge_3.t_f), 5)
    new_p2_time_to_p3_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p2_sampled_set_dict['time_to_p3'].lower_bound, p2_sampled_set_dict['time_to_p3'].upper_bound, p2_sampled_set_dict['time_to_p3'].num_points), optimal_edge_3.t_f), 5)
    new_p2_pos_x_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p2_sampled_set_dict['pos_x'].lower_bound, p2_sampled_set_dict['pos_x'].upper_bound, p2_sampled_set_dict['pos_x'].num_points), optimal_node_2.position[0]), 5)
    new_p2_pos_y_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p2_sampled_set_dict['pos_y'].lower_bound, p2_sampled_set_dict['pos_y'].upper_bound, p2_sampled_set_dict['pos_y'].num_points), optimal_node_2.position[1]), 5)
    new_p2_pos_z_sampled_set = SampledSet(*get_nearest_array_neighbors(np.linspace(p2_sampled_set_dict['pos_z'].lower_bound, p2_sampled_set_dict['pos_z'].upper_bound, p2_sampled_set_dict['pos_z'].num_points), optimal_node_2.position[2]), 5)

    new_p1_sampled_set_dict = {
        'time': new_p1_time_sampled_set,
        'pos_x': new_p1_pos_x_sampled_set,
        'pos_y': new_p1_pos_y_sampled_set,
        'pos_z': new_p1_pos_z_sampled_set
    }

    new_p2_sampled_set_dict = {
        'time': new_p2_time_sampled_set,
        'time_to_p3': new_p2_time_to_p3_sampled_set,
        'pos_x': new_p2_pos_x_sampled_set,
        'pos_y': new_p2_pos_y_sampled_set,
        'pos_z': new_p2_pos_z_sampled_set
    }


    output = OutputData(start_node, optimal_node_1, optimal_node_2, optimal_node_3, optimal_edge_1, optimal_edge_2, optimal_edge_3, new_p1_sampled_set_dict, new_p2_sampled_set_dict, total_edges, total_valid_edges, end_masses)
    return output





initial_r = np.array([200.0, 30.0, 0.0])
initial_v = np.array([-6.0, -0.0, 0.0])
initial_a = np.array([-3.7, -4.0, 0.0])
final_a = np.array([1.0, -0.25, 0.0])

lander = Vehicle(2000, 1000, 10000, 3000, 300)

initial_r = np.array([500.0, 50.0, 0.0])
initial_v = np.array([-50.0, -0.0, 0.0])
initial_a = np.array([-9.8, -8.0, 0.0])
final_a = np.array([1.0, -0.0, 0.0])

lander = Vehicle(150000, 100000, 9000000, 1000000, 320)

p1_time_sampled_set = SampledSet(0.1, 20, 6)
p1_pos_x_sampled_set = SampledSet(0.51*initial_r[0], initial_r[0], 4)
p1_pos_y_sampled_set = SampledSet(-50, 50.0, 5)
p1_pos_z_sampled_set = SampledSet(-50, 50.0, 5)

p2_time_sampled_set = SampledSet(0.1, 20, 6)
p2_time_to_p3_sampled_set = SampledSet(0.1, 20, 6)
p2_pos_x_sampled_set = SampledSet(0.01*initial_r[0], 0.50*initial_r[0], 4)
p2_pos_y_sampled_set = SampledSet(-50, 50.0, 5)
p2_pos_z_sampled_set = SampledSet(-50, 50.0, 5)

p1_sampled_set_dict = {
    'time': p1_time_sampled_set,
    'pos_x': p1_pos_x_sampled_set,
    'pos_y': p1_pos_y_sampled_set,
    'pos_z': p1_pos_z_sampled_set
}

p2_sampled_set_dict = {
    'time': p2_time_sampled_set,
    'time_to_p3': p2_time_to_p3_sampled_set,
    'pos_x': p2_pos_x_sampled_set,
    'pos_y': p2_pos_y_sampled_set,
    'pos_z': p2_pos_z_sampled_set
}

guidance_output = generate_stitcher_trajectory_constant_accel(lander, initial_r, initial_v, initial_a, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), final_a, p1_sampled_set_dict, p2_sampled_set_dict)

# guidance_output = generate_stitcher_trajectory_constant_accel(lander, initial_r, initial_v, initial_a, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), final_a, guidance_output.p1_nearest_neighbors_dict, guidance_output.p2_nearest_neighbors_dict)

# guidance_output = generate_stitcher_trajectory_constant_accel(lander, initial_r, initial_v, initial_a, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), final_a, guidance_output.p1_nearest_neighbors_dict, guidance_output.p2_nearest_neighbors_dict)

# guidance_output = generate_stitcher_trajectory_constant_accel(lander, initial_r, initial_v, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), guidance_output.p1_nearest_neighbors_dict, guidance_output.p2_nearest_neighbors_dict)


print(guidance_output.total_valid_edges)
print(guidance_output.total_valid_edges/guidance_output.total_edges)
print(sum(guidance_output.end_masses)/len(guidance_output.end_masses))
print(max(guidance_output.end_masses))
print(min(guidance_output.end_masses))
print(len(guidance_output.end_masses))


print('Optimal times:')
print(guidance_output.optimal_edge_1.t_f, guidance_output.optimal_edge_2.t_f, guidance_output.optimal_edge_3.t_f)
print('Optimal positions:')
print(guidance_output.start_node.position, guidance_output.optimal_node_1.position, guidance_output.optimal_node_2.position, guidance_output.optimal_node_3.position)
print('Optimal velocities:')
print(guidance_output.start_node.velocity, guidance_output.optimal_node_1.velocity, guidance_output.optimal_node_2.velocity, guidance_output.optimal_node_3.velocity)
print('Optimal accels:')
print(np.linalg.norm(guidance_output.optimal_edge_1.c_0_array), np.linalg.norm(guidance_output.optimal_edge_2.c_0_array), np.linalg.norm(guidance_output.optimal_edge_3.c_0_array))
print(np.linalg.norm(guidance_output.optimal_edge_1.c_0_array + guidance_output.optimal_edge_1.c_1_array*guidance_output.optimal_edge_1.t_f), np.linalg.norm(guidance_output.optimal_edge_2.c_0_array + guidance_output.optimal_edge_2.c_1_array*guidance_output.optimal_edge_2.t_f), np.linalg.norm(guidance_output.optimal_edge_3.c_0_array + guidance_output.optimal_edge_3.c_1_array*guidance_output.optimal_edge_3.t_f))
print(guidance_output.optimal_edge_1.c_0_array, guidance_output.optimal_edge_2.c_0_array, guidance_output.optimal_edge_3.c_0_array)
print(guidance_output.optimal_edge_1.c_0_array + guidance_output.optimal_edge_1.c_1_array*guidance_output.optimal_edge_1.t_f, guidance_output.optimal_edge_2.c_0_array + guidance_output.optimal_edge_2.c_1_array*guidance_output.optimal_edge_2.t_f, guidance_output.optimal_edge_3.c_0_array + guidance_output.optimal_edge_3.c_1_array*guidance_output.optimal_edge_3.t_f)
print('Optimal mass consumed:')
print(guidance_output.optimal_edge_1.mass_consumed, guidance_output.optimal_edge_2.mass_consumed, guidance_output.optimal_edge_3.mass_consumed)
print('Total mass consumed:' + str(round(lander.wet_mass - guidance_output.optimal_edge_3.end_mass, 2)) + ' kg, end vehicle mass: ' + str(round(guidance_output.optimal_edge_3.end_mass, 2)) + ' kg')




t_p_1 = np.linspace(0, guidance_output.optimal_edge_1.t_f, 100)
r_x_p_1 = guidance_output.start_node.position[0] + guidance_output.start_node.velocity[0]*t_p_1 + 0.5*(guidance_output.optimal_edge_1.c_0_array[0] - planetary_body_config.body_surface_gravity)*t_p_1**2 + 1.0/6.0*guidance_output.optimal_edge_1.c_1_array[0]*t_p_1**3
r_y_p_1 = guidance_output.start_node.position[1] + guidance_output.start_node.velocity[1]*t_p_1 + 0.5*(guidance_output.optimal_edge_1.c_0_array[1])*t_p_1**2 + 1.0/6.0*guidance_output.optimal_edge_1.c_1_array[1]*t_p_1**3
r_z_p_1 = guidance_output.start_node.position[2] + guidance_output.start_node.velocity[2]*t_p_1 + 0.5*(guidance_output.optimal_edge_1.c_0_array[2])*t_p_1**2 + 1.0/6.0*guidance_output.optimal_edge_1.c_1_array[2]*t_p_1**3

t_p_2 = np.linspace(0.0, guidance_output.optimal_edge_2.t_f, 100)
r_x_p_2 = guidance_output.optimal_node_1.position[0] + guidance_output.optimal_node_1.velocity[0]*t_p_2 + 0.5*(guidance_output.optimal_edge_2.c_0_array[0] - planetary_body_config.body_surface_gravity)*t_p_2**2 + 1.0/6.0*guidance_output.optimal_edge_2.c_1_array[0]*t_p_2**3
r_y_p_2 = guidance_output.optimal_node_1.position[1] + guidance_output.optimal_node_1.velocity[1]*t_p_2 + 0.5*(guidance_output.optimal_edge_2.c_0_array[1])*t_p_2**2 + 1.0/6.0*guidance_output.optimal_edge_2.c_1_array[1]*t_p_2**3
r_z_p_2 = guidance_output.optimal_node_1.position[2] + guidance_output.optimal_node_1.velocity[2]*t_p_2 + 0.5*(guidance_output.optimal_edge_2.c_0_array[2])*t_p_2**2 + 1.0/6.0*guidance_output.optimal_edge_2.c_1_array[2]*t_p_2**3

t_p_3 = np.linspace(0.0, guidance_output.optimal_edge_3.t_f, 100)
r_x_p_3 = guidance_output.optimal_node_2.position[0] + guidance_output.optimal_node_2.velocity[0]*t_p_3 + 0.5*(guidance_output.optimal_edge_3.c_0_array[0] - planetary_body_config.body_surface_gravity)*t_p_3**2 + 1.0/6.0*guidance_output.optimal_edge_3.c_1_array[0]*t_p_3**3
r_y_p_3 = guidance_output.optimal_node_2.position[1] + guidance_output.optimal_node_2.velocity[1]*t_p_3 + 0.5*(guidance_output.optimal_edge_3.c_0_array[1])*t_p_3**2 + 1.0/6.0*guidance_output.optimal_edge_3.c_1_array[1]*t_p_3**3
r_z_p_3 = guidance_output.optimal_node_2.position[2] + guidance_output.optimal_node_2.velocity[2]*t_p_3 + 0.5*(guidance_output.optimal_edge_3.c_0_array[2])*t_p_3**2 + 1.0/6.0*guidance_output.optimal_edge_3.c_1_array[2]*t_p_3**3

t_plotting = np.concatenate((t_p_1, t_p_2 + guidance_output.optimal_edge_1.t_f, t_p_3 + guidance_output.optimal_edge_1.t_f + guidance_output.optimal_edge_2.t_f))
r_x_plotting = np.concatenate((r_x_p_1, r_x_p_2, r_x_p_3))
r_y_plotting = np.concatenate((r_y_p_1, r_y_p_2, r_y_p_3))
r_z_plotting = np.concatenate((r_z_p_1, r_z_p_2, r_z_p_3))

plotting_functions.plot_3d_data(t_plotting, -r_z_plotting, r_y_plotting, r_x_plotting)

bins = np.linspace(1920, 1950, 100)
plt.hist(guidance_output.end_masses)
plt.show()