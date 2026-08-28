import numpy as np
import orbital_dynamics
import cartesian_dynamics
import planetary_body_config
import matplotlib.pyplot as plt
import plotting_functions
import animation_functions
import copy
import time
import random


class Vehicle:
    def __init__(self, wet_mass, dry_mass, max_thrust, min_thrust, Isp):
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.Isp = Isp
        self.v_e = Isp * 9.80665


class StartNode:
    def __init__(self, position, velocity, acceleration_dir, time):
        self.position = position
        self.velocity = velocity
        self.acceleration_dir = acceleration_dir / np.linalg.norm(acceleration_dir)
        self.time = time
        self.target_edges = []
        self.parent_edges = []
    
    
    def create_and_append_edge(self, target_node):
        t_f = target_node.time

        c_0_array = target_node.acceleration_p1_start.copy()
        c_0_array[0] -= planetary_body_config.body_surface_gravity
        c_1_array = 6.0 / t_f**3 * (target_node.position - self.position - self.velocity * t_f - 0.5*t_f**2 * c_0_array)
        v_f = self.velocity + c_0_array * t_f + 0.5*t_f**2 * c_1_array
        c_0_array[0] += planetary_body_config.body_surface_gravity
        target_node.update_node_velocity(v_f)
        target_node.update_node_acceleration(c_0_array + c_1_array * t_f)

        new_edge = Edge(self, target_node, c_0_array, c_1_array, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), t_f)
        self.target_edges.append(new_edge)
        target_node.parent_edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

class P1Node:
    def __init__(self, position, velocity, acceleration, acceleration_p1_start, time):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.acceleration_p1_start = acceleration_p1_start
        self.time = time
        self.target_edges = []
        self.parent_edges = []
    
    
    def create_and_append_edge(self, target_node):
        t_f = target_node.time

        time_matrix_inverse = np.linalg.inv(np.array([[t_f, t_f**2, t_f**3], 
                                                      [1.0/2.0*t_f**2, 1.0/3.0*t_f**3, 1.0/4.0*t_f**4], 
                                                      [1.0/6.0*t_f**3, 1.0/12.0*t_f**4, 1.0/20.0*t_f**5]]))
        # need to remove gravity accel from self.accel and target.accels first # coefficient_array = time_matrix_inverse @ np.array([target_node.acceleration - self.acceleration, target_node.velocity - self.velocity - self.acceleration * t_f, target_node.position - self.position - self.velocity * t_f - 1.0/2.0*self.acceleration*t_f**2]).T
        target_accel_propagation = target_node.acceleration.copy()
        target_accel_propagation[0] -= planetary_body_config.body_surface_gravity
        self_accel_propagation = self.acceleration.copy()
        self_accel_propagation[0] -= planetary_body_config.body_surface_gravity

        coefficient_array = time_matrix_inverse @ np.array([target_accel_propagation - self_accel_propagation, target_node.velocity - self.velocity - self_accel_propagation * t_f, target_node.position - self.position - self.velocity * t_f - 1.0/2.0*self_accel_propagation*t_f**2])
        c_0_array = self.acceleration.copy()
        c_1_array = coefficient_array[0]
        c_2_array = coefficient_array[1]
        c_3_array = coefficient_array[2]

        if np.linalg.norm(np.abs((c_0_array + c_1_array*t_f + c_2_array*t_f**2 + c_3_array*t_f**3) - target_node.acceleration)) > 0.001:
            breakpoint()

        new_edge = Edge(self, target_node, c_0_array, c_1_array, c_2_array, c_3_array, t_f)
        self.target_edges.append(new_edge)
        target_node.parent_edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

    def update_node_acceleration(self, input_acceleration):
        self.acceleration = input_acceleration

class P2Node:
    def __init__(self, position, velocity, acceleration, acceleration_p3_end, time, time_to_p3):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.acceleration_p3_end = acceleration_p3_end
        self.time = time
        self.time_to_p3 = time_to_p3
        self.target_edges = []
        self.parent_edges = []
    
    def create_and_append_backward_edge(self, target_node):
        t_f = self.time_to_p3

        global_end_acceleration = self.acceleration_p3_end.copy()
        global_end_acceleration[0] -= planetary_body_config.body_surface_gravity
        c_0_array = -2.0 * global_end_acceleration - 6.0 / t_f**2 * (target_node.position - self.position - target_node.velocity*t_f)
        c_1_array = 3.0 / t_f * global_end_acceleration + 6 / t_f**3 * (target_node.position - self.position - target_node.velocity*t_f)
        v_0 = target_node.velocity - c_0_array*t_f - 0.5*t_f**2 * c_1_array
        c_0_array[0] += planetary_body_config.body_surface_gravity
        self.update_node_velocity(v_0)
        self.update_node_acceleration(c_0_array)

        new_edge = Edge(self, target_node, c_0_array, c_1_array, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), t_f)
        self.target_edges.append(new_edge)
        target_node.parent_edges.append(new_edge)
        return new_edge
    
    def update_node_velocity(self, input_velocity):
        self.velocity = input_velocity

    def update_node_acceleration(self, input_acceleration):
        self.acceleration = input_acceleration

class P3Node:
    def __init__(self, position, velocity, acceleration_dir):
        self.position = position
        self.velocity = velocity
        self.acceleration_dir = acceleration_dir / np.linalg.norm(acceleration_dir)
        self.target_edges = []
        self.parent_edges = []




class Edge:
    def __init__(self, parent_node, target_node, c_0_array, c_1_array, c_2_array, c_3_array, t_f):
        self.parent_node = parent_node
        self.target_node = target_node
        self.c_0_array = c_0_array
        self.c_1_array = c_1_array
        self.c_2_array = c_2_array
        self.c_3_array = c_3_array
        self.t_f = t_f
        self.mass_consumed = None
        self.end_mass = None
        self.num_eval_points = 25
        self.thrust_mag_array = None

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

    def check_linear_accel_thrust_bounds(self, edge_start_mass, v_e, thrust_bound_lower, thrust_bound_upper, mass_consumed_flag=True):
        # only checks start and end, but should probably also check a few points in between
        start_accel_mag = np.linalg.norm(self.c_0_array)
        end_accel_mag = np.linalg.norm(self.c_0_array + self.c_1_array*self.t_f)
        mid_accel_mag = np.linalg.norm(self.c_0_array + self.c_1_array*self.t_f/2.0)
        avg_accel_mag = 1.0/3.0*(start_accel_mag + mid_accel_mag + end_accel_mag)
        temp_thrust_integral = edge_start_mass * v_e * (1 - np.exp(-avg_accel_mag* self.t_f / v_e))
        temp_mass_consumed = temp_thrust_integral / v_e
        if not mass_consumed_flag:
            temp_mass_consumed = 0.0
        start_thrust_mag = start_accel_mag * edge_start_mass
        mid_thrust_mag = mid_accel_mag * (edge_start_mass - temp_mass_consumed/2.0)
        end_thrust_mag = end_accel_mag * (edge_start_mass - temp_mass_consumed)
        # print(self.c_0_array[0], (self.c_0_array + self.c_1_array*self.t_f)[0], self.parent_node.position[0], self.parent_node.velocity[0], self.target_node.position[0], self.target_node.velocity[0], self.t_f)
        if start_thrust_mag > thrust_bound_upper or start_thrust_mag < thrust_bound_lower or end_thrust_mag > thrust_bound_upper or end_thrust_mag < thrust_bound_lower or mid_thrust_mag > thrust_bound_upper or mid_thrust_mag < thrust_bound_lower:
            # if not mass_consumed_flag:
            #     print(start_thrust_mag / thrust_bound_upper, end_thrust_mag / thrust_bound_upper)
            return False
        return True

    def compute_cubic_accel_mass_consumed(self, edge_start_mass, v_e):


        temp_mass = edge_start_mass
        for pt in range(self.num_eval_points - 1):
            temp_mass_consumed = 0.5 * (self.thrust_mag_array[pt] + self.thrust_mag_array[pt+1]) * self.t_f/(self.num_eval_points - 1.0) * 1/v_e
            temp_mass -= temp_mass_consumed
            # consider updating thrust_mag_array here to refine
        self.end_mass = temp_mass
        self.mass_consumed = edge_start_mass - temp_mass


    def check_cubic_accel_thrust_bounds(self, edge_start_mass, v_e, thrust_bound_lower, thrust_bound_upper, mass_consumed_flag=True):

        if self.thrust_mag_array is None:
            thrust_mag_array = []
            temp_mass = edge_start_mass
            for pt in range(self.num_eval_points):
                accel = np.linalg.norm(self.c_0_array + self.c_1_array*(pt*self.t_f/(self.num_eval_points - 1.0)) + self.c_2_array*(pt*self.t_f/(self.num_eval_points - 1.0))**2 + self.c_3_array*(pt*self.t_f/(self.num_eval_points - 1.0))**3)
                thrust = temp_mass * accel
                if thrust > thrust_bound_upper or thrust < thrust_bound_lower:
                    return False
                temp_mass_consumed = thrust * self.t_f/(self.num_eval_points - 1.0) * 1/v_e
                temp_mass -= temp_mass_consumed
                thrust_mag_array.append(thrust)

            self.thrust_mag_array = np.asarray(thrust_mag_array)

        else:
            # check midpoint first to speed things up
            if self.thrust_mag_array[self.num_eval_points // 2] > thrust_bound_upper or self.thrust_mag_array[self.num_eval_points // 2] < thrust_bound_lower:
                return False
            for thrust in np.delete(self.thrust_mag_array, self.num_eval_points // 2):
                if thrust > thrust_bound_upper or thrust < thrust_bound_lower:
                    return False
        return True

    def check_position_keepout(self, constraints):

        t_p_2 = np.linspace(0.0, self.t_f, 20)
        r_x_p_2 = self.parent_node.position[0] + self.parent_node.velocity[0]*t_p_2 + 0.5*(self.c_0_array[0] - planetary_body_config.body_surface_gravity)*t_p_2**2 + 1.0/6.0*self.c_1_array[0]*t_p_2**3 + 1.0/12.0*self.c_2_array[0]*t_p_2**4 + 1.0/20.0*self.c_3_array[0]*t_p_2**5
        r_y_p_2 = self.parent_node.position[1] + self.parent_node.velocity[1]*t_p_2 + 0.5*(self.c_0_array[1])*t_p_2**2 + 1.0/6.0*self.c_1_array[1]*t_p_2**3 + 1.0/12.0*self.c_2_array[1]*t_p_2**4 + 1.0/20.0*self.c_3_array[1]*t_p_2**5
        r_z_p_2 = self.parent_node.position[2] + self.parent_node.velocity[2]*t_p_2 + 0.5*(self.c_0_array[2])*t_p_2**2 + 1.0/6.0*self.c_1_array[2]*t_p_2**3 + 1.0/12.0*self.c_2_array[2]*t_p_2**4 + 1.0/20.0*self.c_3_array[2]*t_p_2**5
        for time_idx in range(len(t_p_2)):
            if r_x_p_2[time_idx] < constraints.position_keepout_max_x and r_x_p_2[time_idx] > constraints.position_keepout_min_x:
                if (r_y_p_2[time_idx] < constraints.position_keepout_max_y and r_y_p_2[time_idx] > constraints.position_keepout_min_y) and (r_z_p_2[time_idx] < constraints.position_keepout_max_z and r_z_p_2[time_idx] > constraints.position_keepout_min_z):
                    return False
        return True



class SampledSet:
    def __init__(self, lower_bound, upper_bound, num_points):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_points = num_points

class Constraints:
    def __init__(self, glideslope_max_angle=None, thrust_max_angle=None, interphase_max_angle=None, position_keepout_coords=None):
        self.glideslope_max_angle = glideslope_max_angle
        self.thrust_max_angle = thrust_max_angle
        self.interphase_max_angle = interphase_max_angle
        self.position_keepout_coords = position_keepout_coords
        if self.position_keepout_coords is not None:
            self.position_keepout_max_x = np.max(position_keepout_coords[:,0])
            self.position_keepout_min_x = np.min(position_keepout_coords[:,0])
            self.position_keepout_max_y = np.max(position_keepout_coords[:,1])
            self.position_keepout_min_y = np.min(position_keepout_coords[:,1])
            self.position_keepout_max_z = np.max(position_keepout_coords[:,2])
            self.position_keepout_min_z = np.min(position_keepout_coords[:,2])
        
class OutputData:
    def __init__(self, start_node, optimal_node_1, optimal_node_2, optimal_node_3, optimal_edge_1, optimal_edge_2, optimal_edge_3, p1_nearest_neighbors_dict, p2_nearest_neighbors_dict, total_edges, total_valid_edges, end_masses, solver_durations, process_plot_outputs=None):
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
        self.solver_durations = solver_durations

        if process_plot_outputs is not None:
            self.initial_plot_output_p1_edges = process_plot_outputs[0]
            self.initial_plot_output_p2_edges = process_plot_outputs[1]
            self.initial_plot_output_p3_edges = process_plot_outputs[2]
            self.valid_plot_output_p1_edges = process_plot_outputs[3]
            self.valid_plot_output_p2_edges = process_plot_outputs[4]
            self.valid_plot_output_p3_edges = process_plot_outputs[5]


import numpy as np

def compute_trajectory_states(t, c0, c1, c2, c3, r0, v0, g):

    t = np.asarray(t)
    c0 = np.asarray(c0)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    c3 = np.asarray(c3)
    r0 = np.asarray(r0)
    v0 = np.asarray(v0)
    
    g_vec = np.array([g, 0.0, 0.0])
    
    t_col = t[:, np.newaxis]
    
    r = (r0 
         + (v0 * t_col) 
         + (0.5 * (c0 - g_vec) * t_col**2) 
         + ((1.0 / 6.0) * c1 * t_col**3) 
         + ((1.0 / 12.0) * c2 * t_col**4) 
         + ((1.0 / 20.0) * c3 * t_col**5))
    
    v = (v0 
         + ((c0 - g_vec) * t_col) 
         + (0.5 * c1 * t_col**2) 
         + ((1.0 / 3.0) * c2 * t_col**3) 
         + ((1.0 / 4.0) * c3 * t_col**4))
    
    u = (c0 
         + (c1 * t_col) 
         + (c2 * t_col**2) 
         + (c3 * t_col**3))
    
    return r, v, u

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

def generate_acceleration_set(acc_mag_and_thrust_level, acc_azimuth, acc_zenith):
    output = []
    for i in range(len(acc_mag_and_thrust_level)):
        for j in range(len(acc_azimuth)):
            for k in range(len(acc_zenith)):
                acc_mag = acc_mag_and_thrust_level[i][0]
                thrust_level = acc_mag_and_thrust_level[i][1]
                acc_z = -acc_mag * np.sin(acc_zenith[k]) * np.cos(acc_azimuth[j])
                acc_y = acc_mag * np.sin(acc_zenith[k]) * np.sin(acc_azimuth[j])
                acc_x = acc_mag * np.cos(acc_zenith[k])
                output.append([np.array([acc_x, acc_y, acc_z]), thrust_level])
    # output = np.asarray(output)
    return output

def generate_phase_1_nodes(vehicle, time_sampled_set, start_node, thrust_level_sampled_state, acc_azimuth_sampled_set, acc_zenith_sampled_set):

    sampled_time_array = np.linspace(time_sampled_set.lower_bound, time_sampled_set.upper_bound, time_sampled_set.num_points)
    sampled_acc_azimuth_array = np.linspace(acc_azimuth_sampled_set.lower_bound, acc_azimuth_sampled_set.upper_bound, acc_azimuth_sampled_set.num_points)
    sampled_acc_zenith_array = np.linspace(acc_zenith_sampled_set.lower_bound, acc_zenith_sampled_set.upper_bound, acc_zenith_sampled_set.num_points)
    sampled_thrust_level_array = np.linspace(thrust_level_sampled_state.lower_bound, thrust_level_sampled_state.upper_bound, thrust_level_sampled_state.num_points)

    phase_1_nodes = []
    for time in sampled_time_array:
        sampled_acc_mag_array = []
        for thrust_level in sampled_thrust_level_array:
            thrust_magnitude = thrust_level*vehicle.max_thrust + (1-thrust_level)*vehicle.min_thrust
            end_mass = vehicle.wet_mass - thrust_magnitude/vehicle.v_e * time
            sampled_acc_mag_array.append([thrust_magnitude / end_mass, thrust_level])
        sampled_accelerations_array = generate_acceleration_set(sampled_acc_mag_array, sampled_acc_azimuth_array, sampled_acc_zenith_array)
        for acc_and_thrust_level in sampled_accelerations_array:
            acc = acc_and_thrust_level[0]
            thrust_level = acc_and_thrust_level[1]
            c_0_array = start_node.acceleration_dir.copy() * (thrust_level*vehicle.max_thrust + (1-thrust_level)*vehicle.min_thrust) / vehicle.wet_mass
            phase_1_start_acceleration = c_0_array.copy()
            c_0_array[0] -= planetary_body_config.body_surface_gravity
            global_end_acceleration = acc.copy()
            global_end_acceleration[0] -= planetary_body_config.body_surface_gravity
            c_1_array = (global_end_acceleration - c_0_array) / time
            pos = start_node.position + start_node.velocity * time + 1.0/2.0 * time**2 * c_0_array + 1.0/6.0 * time**3 * c_1_array
            phase_1_nodes.append(P1Node(pos, None, None, phase_1_start_acceleration, time))
    return phase_1_nodes

def generate_phase_2_nodes(vehicle, time_sampled_set, time_to_p3_sampled_set, end_node, thrust_level_sampled_state, acc_azimuth_sampled_set, acc_zenith_sampled_set):

    sampled_time_array = np.linspace(time_sampled_set.lower_bound, time_sampled_set.upper_bound, time_sampled_set.num_points)
    sampled_time_to_p3_array = np.linspace(time_to_p3_sampled_set.lower_bound, time_to_p3_sampled_set.upper_bound, time_to_p3_sampled_set.num_points)
    sampled_acc_azimuth_array = np.linspace(acc_azimuth_sampled_set.lower_bound, acc_azimuth_sampled_set.upper_bound, acc_azimuth_sampled_set.num_points)
    sampled_acc_zenith_array = np.linspace(acc_zenith_sampled_set.lower_bound, acc_zenith_sampled_set.upper_bound, acc_zenith_sampled_set.num_points)
    sampled_thrust_level_array = np.linspace(thrust_level_sampled_state.lower_bound, thrust_level_sampled_state.upper_bound, thrust_level_sampled_state.num_points)


    phase_2_nodes = []
    for time in sampled_time_array:
        for time_to_p3 in sampled_time_to_p3_array:
            sampled_acc_mag_array = []
            for thrust_level in sampled_thrust_level_array:
                thrust_magnitude = thrust_level*vehicle.max_thrust + (1-thrust_level)*vehicle.min_thrust
                if thrust_level > 0.5:
                    critical_mass = vehicle.wet_mass - thrust_magnitude/vehicle.v_e * time_to_p3
                else:
                    critical_mass = vehicle.dry_mass + thrust_magnitude/vehicle.v_e * time_to_p3
                sampled_acc_mag_array.append([thrust_magnitude / critical_mass, thrust_level])
            sampled_accelerations_array = generate_acceleration_set(sampled_acc_mag_array, sampled_acc_azimuth_array, sampled_acc_zenith_array)
            for acc_and_thrust_level in sampled_accelerations_array:
                acc = acc_and_thrust_level[0]
                thrust_level = acc_and_thrust_level[1]
                global_start_acceleration = acc.copy()
                global_start_acceleration[0] -= planetary_body_config.body_surface_gravity
                c_0_array = global_start_acceleration
                if thrust_level > 0.5:
                    global_end_acceleration = end_node.acceleration_dir.copy() * (thrust_level*vehicle.max_thrust + (1-thrust_level)*vehicle.min_thrust) / vehicle.wet_mass
                else:
                    global_end_acceleration = end_node.acceleration_dir.copy() * (thrust_level*vehicle.max_thrust + (1-thrust_level)*vehicle.min_thrust) / vehicle.dry_mass
                p3_end_acceleration = global_end_acceleration.copy()
                global_end_acceleration[0] -= planetary_body_config.body_surface_gravity
                c_1_array = (global_end_acceleration - c_0_array) / time_to_p3
                pos = end_node.position - end_node.velocity * time_to_p3 + 1.0/2.0 * time_to_p3**2 * c_0_array + 1.0/3.0 * time_to_p3**3 * c_1_array
                phase_2_nodes.append(P2Node(pos, None, None, p3_end_acceleration, time, time_to_p3))

    return phase_2_nodes


def generate_phase_3_nodes(target_r, target_v, final_a):


    phase_3_nodes = [P3Node(target_r, target_v, final_a)]
    
    return phase_3_nodes
 

def prune_edges(input_mass_array, input_v_e, thrust_lower_bound, thrust_upper_bound, input_edges, mode, input_nodes=None, constraints=None, mass_consumed_flag=True):

    input_edges_copy = input_edges[:]
    for edg_idx in range(len(input_edges_copy)):
        edge = input_edges_copy[edg_idx]
        prune_edge_flag = False

        # check all constraints here
        if not edge.check_cubic_accel_thrust_bounds(input_mass_array[edg_idx], input_v_e, thrust_lower_bound, thrust_upper_bound, mass_consumed_flag=mass_consumed_flag):
            prune_edge_flag = True
        elif constraints.glideslope_max_angle is not None:
            pass
        elif constraints.thrust_max_angle is not None:
            pass
        elif constraints.interphase_max_angle is not None:
            if len(edge.parent_node.parent_edges) > 0 and len(edge.target_node.target_edges) > 0:
                start_interphase_angle = np.arccos(np.dot(edge.c_0_array, edge.parent_node.acceleration) / (np.linalg.norm(edge.c_0_array)*np.linalg.norm(edge.parent_node.acceleration)))
                if start_interphase_angle > constraints.interphase_max_angle:
                    prune_edge_flag = True
                end_interphase_angle = np.arccos(np.dot(edge.c_0_array + edge.c_1_array*edge.t_f, edge.target_node.acceleration) / (np.linalg.norm(edge.c_0_array + edge.c_1_array*edge.t_f)*np.linalg.norm(edge.target_node.acceleration)))
                if end_interphase_angle > constraints.interphase_max_angle:
                    prune_edge_flag = True
        elif constraints.position_keepout_coords is not None:
            if not edge.check_position_keepout(constraints):
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



def generate_stitcher_trajectory_linear_cubic_accel(vehicle, initial_r, initial_v, initial_a, final_r, final_v, final_a, constraints, p1_sampled_set_dict, p2_sampled_set_dict, process_plot=False):

    start_node = StartNode(initial_r, initial_v, initial_a, 0.0)

    # create phase 1 nodes

    phase_1_nodes = generate_phase_1_nodes(vehicle, p1_sampled_set_dict['time'], start_node, p1_sampled_set_dict['thrust_state'], p1_sampled_set_dict['acc_azimuth'], p1_sampled_set_dict['acc_zenith'])
    # phase_1_nodes = generate_phase_1_nodes(p1_sampled_set_dict['time'], p1_sampled_set_dict['pos_x'], p1_sampled_set_dict['pos_y'], p1_sampled_set_dict['pos_z'])

    last_solver_checkpoint_time = time.time()
    # connect start and phase 1 nodes with edges
    for phase_1_node in phase_1_nodes:
        start_node.create_and_append_edge(phase_1_node)
    T_p1_edge_generation = time.time() - last_solver_checkpoint_time

    last_solver_checkpoint_time = time.time()
    # prune phase 1 edges and nodes that violate constraints
    prune_edges([vehicle.wet_mass]*len(start_node.target_edges), vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust, start_node.target_edges, 'forward', phase_1_nodes, constraints)
    total_p1_edges = 0
    for phase_1_edge in start_node.target_edges:
        total_p1_edges += 1
    print('Total p1 edges: ' + str(total_p1_edges))

    # compute costs of phase 1 edges
    for start_edge in start_node.target_edges:
        start_edge.compute_cubic_accel_mass_consumed(vehicle.wet_mass, vehicle.v_e)

    total_p1_edges = 0
    for phase_1_edge in start_node.target_edges:
        total_p1_edges += 1


    T_p1_edge_cost_pruning = time.time() - last_solver_checkpoint_time

    if process_plot:
        plot_output_num_edges = 10
        plot_output_p1_edges = random.sample(start_node.target_edges, plot_output_num_edges)

    # create phase 3 nodes
    phase_3_nodes = generate_phase_3_nodes(final_r, final_v, final_a)

    # create phase 2 nodes
    phase_2_nodes = generate_phase_2_nodes(vehicle, p2_sampled_set_dict['time'], p2_sampled_set_dict['time_to_p3'], phase_3_nodes[0], p2_sampled_set_dict['thrust_state'], p2_sampled_set_dict['acc_azimuth'], p2_sampled_set_dict['acc_zenith'])
    # phase_2_nodes = generate_phase_2_nodes(p2_sampled_set_dict['time'], p2_sampled_set_dict['time_to_p3'], p2_sampled_set_dict['pos_x'], p2_sampled_set_dict['pos_y'], p2_sampled_set_dict['pos_z'])



    last_solver_checkpoint_time = time.time()
    # connect phase 3 and phase 2 nodes with edges
    for phase_2_node in phase_2_nodes:
        for phase_3_node in phase_3_nodes:
            phase_2_node.create_and_append_backward_edge(phase_3_node)

    total_p3_edges = 0
    for phase_3_edge in phase_3_nodes[0].parent_edges:
        total_p3_edges += 1
    T_p3_edge_generation = time.time() - last_solver_checkpoint_time

    if process_plot:
        plot_output_num_edges = 10
        plot_output_p3_edges = random.sample(phase_3_node.parent_edges, plot_output_num_edges)

    # prune phase 3 edges that violate constraints based on worst case mass
    last_solver_checkpoint_time = time.time()
    # for phase_3_node in phase_3_nodes:
    #     prune_edges([vehicle.dry_mass]*len(phase_3_node.parent_edges), vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust, phase_3_node.parent_edges, 'backward', phase_2_nodes, constraints, mass_consumed_flag=False)
    total_p3_edges = 0
    for phase_3_edge in phase_3_nodes[0].parent_edges:
        total_p3_edges += 1
    print('Total p3 edges: ' + str(total_p3_edges))
    T_p3_edge_conservative_pruning = time.time() - last_solver_checkpoint_time


    # connect phase 1 and phase 2 nodes with edges
    last_solver_checkpoint_time = time.time()
    for phase_1_node in phase_1_nodes:
        for phase_2_node in phase_2_nodes:
            phase_1_node.create_and_append_edge(phase_2_node)
    T_p2_edge_generation = time.time() - last_solver_checkpoint_time



    if process_plot:
        plot_output_p2_edges = []
        for p1_edge in plot_output_p1_edges:
            for p2_edge in p1_edge.target_node.target_edges:
                if p2_edge.target_node.target_edges[0] in plot_output_p3_edges:
                    plot_output_p2_edges.append(p2_edge)



    last_solver_checkpoint_time = time.time()

    # prune phase 2 edges that violate constraints
    for phase_1_node in phase_1_nodes:
        prune_edges([phase_1_node.parent_edges[0].end_mass]*len(phase_1_node.target_edges), vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust, phase_1_node.target_edges, 'forward', phase_2_nodes, constraints)
    total_p2_edges = 0
    for phase_1_node in phase_1_nodes:
        for phase_2_edge in phase_1_node.target_edges:
            total_p2_edges += 1
    print('Total p2 edges: ' + str(total_p2_edges))

    # compute costs of phase 2 edges
    for phase_1_node in phase_1_nodes:
        for edg_idx in range(len(phase_1_node.target_edges)):
            phase_1_node.target_edges[edg_idx].compute_cubic_accel_mass_consumed(phase_1_node.parent_edges[0].end_mass, vehicle.v_e)

    total_p2_edges = 0
    for phase_1_node in phase_1_nodes:
        for phase_2_edge in phase_1_node.target_edges:
            total_p2_edges += 1

    T_p2_edge_cost_pruning = time.time() - last_solver_checkpoint_time

    for phase_2_node in phase_2_nodes:
        if len(phase_2_node.target_edges) != 1:
            print(len(phase_2_node.target_edges))
            raise ValueError


    if process_plot:
        valid_plot_output_p1_edges = []
        valid_plot_output_p3_edges = []
        for p1_edge in plot_output_p1_edges:
            if p1_edge in start_node.target_edges:
                valid_plot_output_p1_edges.append(p1_edge)
        for p3_edge in plot_output_p3_edges:
            if p3_edge in phase_3_node.parent_edges:
                valid_plot_output_p3_edges.append(p3_edge)

        valid_plot_output_p2_edges = []
        for p1_edge in valid_plot_output_p1_edges:
            for p2_edge in p1_edge.target_node.target_edges:
                if p2_edge.target_node.target_edges[0] in valid_plot_output_p3_edges:
                    valid_plot_output_p2_edges.append(p2_edge)
                    
        # print(len(plot_output_p1_edges))
        # print(len(plot_output_p3_edges))
        # print(len(plot_output_p2_edges))
        # print()
        # print(len(valid_plot_output_p1_edges))
        # print(len(valid_plot_output_p3_edges))
        # print(len(valid_plot_output_p2_edges))
        # breakpoint()


    total_edges = 0
    for phase_1_node in phase_1_nodes:
        for phase_2_edge in phase_1_node.target_edges:
            total_edges += 1

    last_solver_checkpoint_time = time.time()
    end_masses = []
    best_mass = 0.0
    total_valid_edges = 0
    valid_p3_edges = []
    for phase_2_node in phase_2_nodes:
        if len(phase_2_node.target_edges) != 1:
            raise ValueError('should be 1 phase 2 target edge')
        for parent_edge in phase_2_node.parent_edges:
            current_edge_start_mass = parent_edge.end_mass
            target_edge = phase_2_node.target_edges[0]
            # need to have this be a prune_edges call instead
            if target_edge.check_cubic_accel_thrust_bounds(current_edge_start_mass, vehicle.v_e, vehicle.min_thrust, vehicle.max_thrust):
                if target_edge.check_position_keepout(constraints):
                    valid_p3_edges.append(target_edge)
                    target_edge.compute_cubic_accel_mass_consumed(current_edge_start_mass, vehicle.v_e)
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
    T_p3_edge_cost_pruning = time.time() - last_solver_checkpoint_time


    
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


    solver_durations = {'T_p1_edge_generation': T_p1_edge_generation,
                        'T_p1_edge_cost_pruning': T_p1_edge_cost_pruning,
                        'T_p3_edge_generation': T_p3_edge_generation,
                        'T_p3_edge_conservative_pruning': T_p3_edge_conservative_pruning,
                        'T_p2_edge_generation': T_p2_edge_generation, 
                        'T_p2_edge_cost_pruning': T_p2_edge_cost_pruning,
                        'T_p3_edge_cost_pruning': T_p3_edge_cost_pruning}
    



    if process_plot:
        output = OutputData(start_node, optimal_node_1, optimal_node_2, optimal_node_3, optimal_edge_1, optimal_edge_2, optimal_edge_3, new_p1_sampled_set_dict, new_p2_sampled_set_dict, total_edges, total_valid_edges, end_masses, solver_durations, [plot_output_p1_edges, plot_output_p2_edges, plot_output_p3_edges, valid_plot_output_p1_edges, valid_plot_output_p2_edges, valid_plot_output_p3_edges])
    else:
        output = OutputData(start_node, optimal_node_1, optimal_node_2, optimal_node_3, optimal_edge_1, optimal_edge_2, optimal_edge_3, new_p1_sampled_set_dict, new_p2_sampled_set_dict, total_edges, total_valid_edges, end_masses, solver_durations)
    return output





initial_r = np.array([200.0, 30.0, 0.0])
initial_v = np.array([-6.0, -0.0, 0.0])
initial_a = np.array([-3.7, -4.0, 0.0])
final_a = np.array([1.0, -0.25, 0.0])

lander = Vehicle(2000, 1000, 10000, 3000, 300)

# initial_r = np.array([500.0, 100.0, 200.0])
# initial_v = np.array([-80.0, -40.0, -20.0])
# initial_a = np.array([10.0, 8.04, -3.66])
initial_r = np.array([500.0, 0.0, 80.0])
initial_v = np.array([-80.0, 0.0, 0.0])
initial_a = np.array([0.0, 0.0, -1.0])
final_r = np.array([0.0, 0.0, 0.0])
final_v = np.array([0.0, 0.0, 0.0])
final_a = np.array([13.0, -0.0, 0.0])
initial_a = initial_a / np.linalg.norm(initial_a)
final_a = final_a / np.linalg.norm(final_a)

lander = Vehicle(150000, 135000, 6000000, 2000000, 320)
tower_coords = np.array([[0, -20, 5], [0, 20, 5], [0, -20, 45], [0, 20, 45], [120, -20, 5], [120, 20, 5], [120, -20, 45], [120, 20, 45]])
# tower_coords = np.array([[0.0, 0.0, 0.0]])
constraints = Constraints(position_keepout_coords=tower_coords)

tower_plotting_coords = tower_coords
tower_plotting_coords = np.array([[0, -10, 15], [0, 10, 15], [0, -10, 35], [0, 10, 35], [120, -10, 15], [120, 10, 15], [120, -10, 35], [120, 10, 35]])

p1_time_sampled_set = SampledSet(1.5, 10, 6)

p1_thrust_state = SampledSet(0.1, 0.9, 2)
azimuth_v_0 = np.arctan2(initial_v[2], initial_v[1])
initial_los_yz = np.array([final_r[1] - initial_r[1], final_r[2] - initial_r[2]])
if np.cross(initial_los_yz, initial_v)[0] > 0:
    p1_accel_azimuth_sampled_set = SampledSet(azimuth_v_0 - np.pi, azimuth_v_0, 5)
else:
    p1_accel_azimuth_sampled_set = SampledSet(azimuth_v_0, azimuth_v_0 + np.pi, 5)
p1_accel_zenith_sampled_set = SampledSet(0, 60*np.pi/180, 5)

p2_thrust_state = SampledSet(0.1, 0.9, 2)
azimuth_v_0 = np.arctan2(initial_v[2], initial_v[1])
initial_los_yz = np.array([final_r[1] - initial_r[1], final_r[2] - initial_r[2]])
if np.cross(initial_los_yz, initial_v)[0] > 0:
    p2_accel_azimuth_sampled_set = SampledSet(azimuth_v_0, azimuth_v_0 + np.pi, 5)
else:
    p2_accel_azimuth_sampled_set = SampledSet(azimuth_v_0 - np.pi, azimuth_v_0, 5)
p2_accel_zenith_sampled_set = SampledSet(0, 60*np.pi/180, 5)

p1_pos_x_sampled_set = SampledSet(0.51*initial_r[0], initial_r[0], 4)
p1_pos_y_sampled_set = SampledSet(-50, 50.0, 5)
p1_pos_z_sampled_set = SampledSet(-50, 50.0, 5)

p2_time_sampled_set = SampledSet(1.5, 10, 6)
p2_time_to_p3_sampled_set = SampledSet(1.5, 10, 6)
p2_pos_x_sampled_set = SampledSet(0.01*initial_r[0], 0.50*initial_r[0], 4)
p2_pos_y_sampled_set = SampledSet(-50, 50.0, 5)
p2_pos_z_sampled_set = SampledSet(-50, 50.0, 5)

p1_sampled_set_dict = {
    'time': p1_time_sampled_set,
    'pos_x': p1_pos_x_sampled_set,
    'pos_y': p1_pos_y_sampled_set,
    'pos_z': p1_pos_z_sampled_set,
    'thrust_state': p1_thrust_state,
    'acc_azimuth': p1_accel_azimuth_sampled_set,
    'acc_zenith': p1_accel_zenith_sampled_set
}

p2_sampled_set_dict = {
    'time': p2_time_sampled_set,
    'time_to_p3': p2_time_to_p3_sampled_set,
    'pos_x': p2_pos_x_sampled_set,
    'pos_y': p2_pos_y_sampled_set,
    'pos_z': p2_pos_z_sampled_set,
    'thrust_state': p2_thrust_state,
    'acc_azimuth': p2_accel_azimuth_sampled_set,
    'acc_zenith': p2_accel_zenith_sampled_set
}

guidance_output = generate_stitcher_trajectory_linear_cubic_accel(lander, initial_r, initial_v, initial_a, final_r, final_v, final_a, constraints, p1_sampled_set_dict, p2_sampled_set_dict, process_plot=True)



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
for key in [*guidance_output.solver_durations]:
    print(key + ': ' + str(round(guidance_output.solver_durations[key], 3)))


total_time = guidance_output.optimal_edge_1.t_f + guidance_output.optimal_edge_2.t_f +guidance_output.optimal_edge_3.t_f
num_total_points_plotting = 300
num_p1_points_plotting = int(guidance_output.optimal_edge_1.t_f/total_time * num_total_points_plotting)
num_p2_points_plotting = int(guidance_output.optimal_edge_2.t_f/total_time * num_total_points_plotting)
num_p3_points_plotting = num_total_points_plotting - num_p1_points_plotting - num_p2_points_plotting

t_p_1 = np.linspace(0, guidance_output.optimal_edge_1.t_f, num_p1_points_plotting)
r_p_1, v_p_1, u_p_1 = compute_trajectory_states(
    t = t_p_1,
    c0 = guidance_output.optimal_edge_1.c_0_array,
    c1 = guidance_output.optimal_edge_1.c_1_array,
    c2 = guidance_output.optimal_edge_1.c_2_array,
    c3 = guidance_output.optimal_edge_1.c_3_array,
    r0 = guidance_output.start_node.position,
    v0 = guidance_output.start_node.velocity,
    g = planetary_body_config.body_surface_gravity
)
r_x_p_1, r_y_p_1, r_z_p_1 = r_p_1[:, 0], r_p_1[:, 1], r_p_1[:, 2]
v_x_p_1, v_y_p_1, v_z_p_1 = v_p_1[:, 0], v_p_1[:, 1], v_p_1[:, 2]
u_x_p_1, u_y_p_1, u_z_p_1 = u_p_1[:, 0], u_p_1[:, 1], u_p_1[:, 2]

t_p_2 = np.linspace(0.0, guidance_output.optimal_edge_2.t_f, num_p2_points_plotting)
r_p_2, v_p_2, u_p_2 = compute_trajectory_states(
    t = t_p_2,
    c0 = guidance_output.optimal_edge_2.c_0_array,
    c1 = guidance_output.optimal_edge_2.c_1_array,
    c2 = guidance_output.optimal_edge_2.c_2_array,
    c3 = guidance_output.optimal_edge_2.c_3_array,
    r0 = guidance_output.optimal_node_1.position,
    v0 = guidance_output.optimal_node_1.velocity,
    g = planetary_body_config.body_surface_gravity
)
r_x_p_2, r_y_p_2, r_z_p_2 = r_p_2[:, 0], r_p_2[:, 1], r_p_2[:, 2]
v_x_p_2, v_y_p_2, v_z_p_2 = v_p_2[:, 0], v_p_2[:, 1], v_p_2[:, 2]
u_x_p_2, u_y_p_2, u_z_p_2 = u_p_2[:, 0], u_p_2[:, 1], u_p_2[:, 2]

t_p_3 = np.linspace(0.0, guidance_output.optimal_edge_3.t_f, num_p3_points_plotting)
r_p_3, v_p_3, u_p_3 = compute_trajectory_states(
    t = t_p_3,
    c0 = guidance_output.optimal_edge_3.c_0_array,
    c1 = guidance_output.optimal_edge_3.c_1_array,
    c2 = guidance_output.optimal_edge_3.c_2_array,
    c3 = guidance_output.optimal_edge_3.c_3_array,
    r0 = guidance_output.optimal_node_2.position,
    v0 = guidance_output.optimal_node_2.velocity,
    g = planetary_body_config.body_surface_gravity
)
r_x_p_3, r_y_p_3, r_z_p_3 = r_p_3[:, 0], r_p_3[:, 1], r_p_3[:, 2]
v_x_p_3, v_y_p_3, v_z_p_3 = v_p_3[:, 0], v_p_3[:, 1], v_p_3[:, 2]
u_x_p_3, u_y_p_3, u_z_p_3 = u_p_3[:, 0], u_p_3[:, 1], u_p_3[:, 2]

t_plotting = np.concatenate((t_p_1, t_p_2 + guidance_output.optimal_edge_1.t_f, t_p_3 + guidance_output.optimal_edge_1.t_f + guidance_output.optimal_edge_2.t_f))
r_x_plotting = np.concatenate((r_x_p_1, r_x_p_2, r_x_p_3))
r_y_plotting = np.concatenate((r_y_p_1, r_y_p_2, r_y_p_3))
r_z_plotting = np.concatenate((r_z_p_1, r_z_p_2, r_z_p_3))
v_x_plotting = np.concatenate((v_x_p_1, v_x_p_2, v_x_p_3))
v_y_plotting = np.concatenate((v_y_p_1, v_y_p_2, v_y_p_3))
v_z_plotting = np.concatenate((v_z_p_1, v_z_p_2, v_z_p_3))
u_x_plotting = np.concatenate((u_x_p_1, u_x_p_2, u_x_p_3))
u_y_plotting = np.concatenate((u_y_p_1, u_y_p_2, u_y_p_3))
u_z_plotting = np.concatenate((u_z_p_1, u_z_p_2, u_z_p_3))

plotting_functions.plot_3d_data_with_rocket(t_plotting, -r_z_plotting, r_y_plotting, r_x_plotting, t_plotting[::20], -u_z_plotting[::20], u_y_plotting[::20], u_x_plotting[::20], rocket_length=40, rocket_radius=4.5, thrust_scale=1.5, polygon_coords=tower_plotting_coords)

animation_functions.plot_3d_data_with_rocket(t_plotting, -r_z_plotting, r_y_plotting, r_x_plotting, t_plotting, -u_z_plotting, u_y_plotting, u_x_plotting, rocket_length=40, rocket_radius=4.5, thrust_scale=1.5, polygon_coords=tower_plotting_coords, fps=10)

plotting_functions.plot_2d_data([t_plotting, t_plotting, t_plotting], [r_x_plotting, r_y_plotting, r_z_plotting], ['rx', 'ry', 'rz'], 'Sampling-based Position vs Time', 'Time (s)', 'Position (m)')

plotting_functions.plot_2d_data([t_plotting, t_plotting, t_plotting], [v_x_plotting,v_y_plotting, v_z_plotting], ['vx', 'vy', 'vz'], 'Sampling-based Velocity vs Time', 'Time (s)', 'Velocity (m/s)')

plotting_functions.plot_2d_data([t_plotting, t_plotting, t_plotting], [u_x_plotting, u_y_plotting, u_z_plotting], ['ux', 'uy', 'uz'], 'Commanded Accel vs Time', 'Time (s)', 'Acceleration (m/s^2)')

plotting_functions.plot_2d_data([t_plotting], [np.sqrt(u_x_plotting**2 + u_y_plotting**2 + u_z_plotting**2)], [''], 'Commanded Accel Magnitude vs time', 'Time (s)', 'Acceleration (m/s^2)')

bins = np.linspace(1920, 1950, 100)
plt.hist(guidance_output.end_masses)
# plt.show()