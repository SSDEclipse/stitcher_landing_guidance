import numpy as np
import orbital_conversions
import orbital_dynamics
import cartesian_dynamics
import planetary_body_config
import matplotlib.pyplot as plt


class Vehicle:
    def __init__(self, wet_mass, dry_mass, max_thrust, min_thrust, Isp):
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.Isp = Isp
        self.v_e = Isp * 9.80665


class StartNode:
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


        c_0_array = np.array([0.0, 0.0, 0.0])
        if np.abs(target_node.position[0] - self.position[0]) > 0.000001:
            c_0_array[0] = 0.5 / (target_node.position[0] - self.position[0]) * (target_node.velocity[0]**2 - self.velocity[0]**2)
        if np.abs(target_node.position[1] - self.position[1]) > 0.000001:
            c_0_array[1] = 0.5 / (target_node.position[1] - self.position[1]) * (target_node.velocity[1]**2 - self.velocity[1]**2)
        if np.abs(target_node.position[2] - self.position[2]) > 0.000001:
            c_0_array[2] = 0.5 / (target_node.position[2] - self.position[2]) * (target_node.velocity[2]**2 - self.velocity[2]**2)
        
        if np.linalg.norm(c_0_array) < 0.0001:
            t_f = 0.0
        else:
            for idx in range(3):
                if np.abs(c_0_array[idx]) > 0.000001:
                    t_f = (target_node.velocity[idx] - self.velocity[idx]) / c_0_array[idx]

        if t_f > 0:
            c_0_array[0] += planetary_body_config.body_surface_gravity
            new_edge = Edge(self, target_node, c_0_array, np.array([0.0, 0.0, 0.0]), t_f)
            self.target_edges.append(new_edge)
            target_node.parent_edges.append(new_edge)
            return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

class P2Node:
    def __init__(self, position, velocity, time_to_p3):
        self.position = position
        self.velocity = velocity
        self.time_to_p3 = time_to_p3
        self.target_edges = []
        self.parent_edges = []
    
    
    def create_and_append_forward_edge(self, target_node):
        t_f = self.time_to_p3
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
    
    def create_and_append_backward_edge(self, target_node):
        t_f = self.time_to_p3
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

class P3Node:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.target_edges = []
        self.parent_edges = []




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





def generate_position_set(pos_x, pos_y, pos_z):
    output = []
    for i in range(len(pos_x)):
        for j in range(len(pos_y)):
            for k in range(len(pos_z)):
                output.append(np.array([pos_x[i], pos_y[j], pos_z[j]]))
    output = np.asarray(output)
    return output

def generate_phase_1_nodes(initial_r):

    sampled_time_array = np.linspace(0.1, 10, 10)
    sampled_position_x_array = np.linspace(0.67*initial_r[0], initial_r[0], 5)
    sampled_position_y_array = np.linspace(0, initial_r[1], 5)
    sampled_position_z_array = np.linspace(0, initial_r[2], 5)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_1_nodes = []
    for time in sampled_time_array:
        for pos in sampled_positions_array:
            phase_1_nodes.append(P1Node(pos, None, time))
    return phase_1_nodes

def generate_phase_2_nodes(initial_r):

    sampled_time_array = np.linspace(0.1, 10, 10)
    sampled_position_x_array = np.linspace(0.34*initial_r[0], 0.66*initial_r[0], 5)
    sampled_position_y_array = np.linspace(0, initial_r[1], 5)
    sampled_position_z_array = np.linspace(0, initial_r[2], 5)
    sampled_positions_array = generate_position_set(sampled_position_x_array, sampled_position_y_array, sampled_position_z_array)

    phase_2_nodes = []
    for time in sampled_time_array:
        for pos in sampled_positions_array:
            phase_2_nodes.append(P2Node(pos, None, time))
    return phase_2_nodes

def generate_phase_3_nodes(target_r, target_v):


    phase_3_nodes = [P3Node(target_r, target_v)]
    
    return phase_3_nodes


def prune_edge(edge, input_mass, thrust_lower_bound, thrust_upper_bound):
        # check all constraints here
        if not edge.check_const_accel_thrust_bounds(input_mass, thrust_lower_bound, thrust_upper_bound):
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

def prune_edges(input_mass_array, thrust_lower_bound, thrust_upper_bound, input_edges, mode, input_nodes=None):

    input_edges_copy = input_edges[:]
    for edg_idx in range(len(input_edges_copy)):
        edge = input_edges_copy[edg_idx]
        prune_edge_flag = False

        # check all constraints here
        if not edge.check_const_accel_thrust_bounds(input_mass_array[edg_idx], thrust_lower_bound, thrust_upper_bound):
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
        edge.compute_const_accel_mass_consumed(v_e)





def generate_stitcher_trajectory_constant_accel(vehicle, initial_r, initial_v, final_r, final_v):

    start_node = StartNode(initial_r, initial_v, 0.0)

    # create phase 1 nodes

    phase_1_nodes = generate_phase_1_nodes(initial_r)

    # connect start and phase 1 nodes with edges
    for phase_1_node in phase_1_nodes:
        start_node.create_and_append_edge(phase_1_node)

    # compute costs of phase 1 edges
    for start_edge in start_node.target_edges:
        start_edge.compute_const_accel_mass_consumed(vehicle.wet_mass, vehicle.v_e)

    # prune phase 1 edges and nodes that violate constraints
    prune_edges([vehicle.wet_mass]*len(start_node.target_edges), vehicle.min_thrust, vehicle.max_thrust, start_node.target_edges, 'forward', phase_1_nodes)
    
    # create phase 3 nodes
    phase_3_nodes = generate_phase_3_nodes(final_r, final_v)

    # create phase 2 nodes
    phase_2_nodes = generate_phase_2_nodes(initial_r)


    # connect phase 3 and phase 2 nodes with edges
    for phase_2_node in phase_2_nodes:
        for phase_3_node in phase_3_nodes:
            phase_2_node.create_and_append_backward_edge(phase_3_node)
            # print(phase_2_node.velocity, phase_3_node.time)

    # prune phase 3 edges that violate constraints based on worst case mass
    for phase_3_node in phase_3_nodes:
        prune_edges([vehicle.dry_mass]*len(phase_3_node.parent_edges), vehicle.min_thrust, vehicle.max_thrust, phase_3_node.parent_edges, 'backward', phase_2_nodes)

    # connect phase 1 and phase 2 nodes with edges
    for phase_1_node in phase_1_nodes:
        for phase_2_node in phase_2_nodes:
            phase_1_node.create_and_append_edge(phase_2_node)


    # compute costs of phase 2 edges
    for phase_1_node in phase_1_nodes:
        for edg_idx in range(len(phase_1_node.target_edges)):
            phase_1_node.target_edges[edg_idx].compute_const_accel_mass_consumed(phase_1_node.parent_edges[0].end_mass, vehicle.v_e)

    # prune phase 2 edges that violate constraints
    for phase_1_node in phase_1_nodes:
        prune_edges([phase_1_node.parent_edges[0].end_mass]*len(phase_1_node.target_edges), vehicle.min_thrust, vehicle.max_thrust, phase_1_node.target_edges, 'forward', phase_2_nodes)
    

    # compute costs of phase 3 edges
    # for phase_2_node in phase_2_nodes:
    #     for edg_idx in range(len(phase_2_node.target_edges)):
    #         phase_2_node.target_edges[edg_idx].compute_const_accel_mass_consumed(phase_2_node.parent_edges[edg_idx].end_mass, vehicle.v_e)



    
    # reprune phase 3 edges that violate constraints based on actual mass
    # for phase_2_node in phase_2_nodes:
    #     phase_2_pruning_mass_array = []
    #     if len(phase_2_node.target_edges) != len(phase_2_node.parent_edges):
    #         print(len(phase_2_node.target_edges), len(phase_2_node.parent_edges))
    #     for mass_idx in range(len(phase_2_node.target_edges)):
    #         phase_2_pruning_mass_array.append(phase_2_node.parent_edges[mass_idx].end_mass)
    #     prune_edges(phase_2_pruning_mass_array, vehicle.min_thrust, vehicle.max_thrust, phase_2_node.target_edges, 'forward', phase_3_nodes)

    for phase_2_node in phase_2_nodes:
        if len(phase_2_node.target_edges) != 1:
            print(len(phase_2_node.target_edges))
            raise ValueError
    print(len(phase_3_nodes[0].parent_edges))   
        

    end_masses = []
    best_mass = 0.0
    counter = 0
    for phase_2_node in phase_2_nodes:
        for parent_edge in phase_2_node.parent_edges:
            current_edge_start_mass = parent_edge.end_mass
            for target_edge in phase_2_node.target_edges: # should only be 1 p2 target edge
                if target_edge.check_const_accel_thrust_bounds(current_edge_start_mass, vehicle.min_thrust, vehicle.max_thrust):
                    target_edge.compute_const_accel_mass_consumed(current_edge_start_mass, vehicle.v_e)
                    counter += 1
                    touchdown_mass = target_edge.end_mass
                    end_masses.append(touchdown_mass)
                    if touchdown_mass > best_mass:
                        best_mass = touchdown_mass
                        optimal_node_3 = phase_3_nodes[0]
                        optimal_node_2 = phase_2_node
                        optimal_node_1 = parent_edge.parent_node
                        optimal_edge_3 = target_edge
                        optimal_edge_2 = parent_edge
                        optimal_edge_1 = optimal_node_1.parent_edges[0]
    print(counter)






    # counter = 0

    # best_mass = 0.0
    # end_masses = []

    # for phase_1_edge in start_node.target_edges:
    #     phase_2_edges = phase_1_edge.target_node.target_edges
    #     # print(len(phase_2_edges))
    #     for phase_2_edge in phase_2_edges:
    #         phase_3_edges = phase_2_edge.target_node.target_edges
    #         # print(len(phase_3_edges))
    #         for phase_3_edge in phase_3_edges:
    #             counter += 1
    #             end_mass = phase_3_edge.end_mass
    #             if end_mass > best_mass:
    #                 optimal_node_1 = phase_1_edge.target_node
    #                 optimal_node_2 = phase_2_edge.target_node
    #                 optimal_node_3 = phase_3_edge.target_node
    #                 optimal_edge_1 = phase_1_edge
    #                 optimal_edge_2 = phase_2_edge
    #                 optimal_edge_3 = phase_3_edge
    #                 best_mass = end_mass
    #             end_masses.append(phase_3_edge.end_mass)
    
    # end_masses = []
    # for phase_2_node in phase_2_nodes:
    #     for edge in phase_2_node.target_edges:
    #         end_masses.append(edge.end_mass)

                
    
    # print(counter)
    print(sum(end_masses)/len(end_masses))
    print(max(end_masses))
    print(min(end_masses))
    print(len(end_masses))


    print('Optimal times:')
    print(optimal_edge_1.t_f, optimal_edge_2.t_f, optimal_edge_3.t_f)
    print('Optimal positions:')
    print(start_node.position, optimal_node_1.position, optimal_node_2.position, optimal_node_3.position)
    print('Optimal velocities:')
    print(start_node.velocity, optimal_node_1.velocity, optimal_node_2.velocity, optimal_node_3.velocity)
    print('Optimal accels:')
    print(optimal_edge_1.c0_array, optimal_edge_2.c0_array, optimal_edge_3.c0_array)
    print(np.linalg.norm(optimal_edge_1.c0_array), np.linalg.norm(optimal_edge_2.c0_array), np.linalg.norm(optimal_edge_3.c0_array))
    print('Optimal mass consumed:')
    print(optimal_edge_1.mass_consumed, optimal_edge_2.mass_consumed, optimal_edge_3.mass_consumed)

    bins = np.linspace(1900, 1980, 100)
    plt.hist(end_masses, bins=bins)
    plt.show()




lander = Vehicle(2000, 1000, 10000, 3000, 300)

generate_stitcher_trajectory_constant_accel(lander, np.array([50.0, 0.0, 0.0]), np.array([-5.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))