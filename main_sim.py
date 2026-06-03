import numpy as np
import orbital_conversions
import orbital_dynamics
import cartesian_dynamics
import planetary_body_config
import constants_and_helpers
import sim_configs
import data_parser
import guidance


def run_sim_mci():

    sim_dt = sim_configs.sim_dt
    sim_duration = sim_configs.sim_duration
    num_iterations = round(sim_duration/sim_dt)


    time = 0
    mass = sim_configs.wet_mass
    r_mci = sim_configs.initial_r_mci
    v_mci = sim_configs.initial_v_mci

    thrust_vector_mci = np.array([0.0, 0.0, 0.0])

    data_logging_time = [time]
    data_logging_mass = [mass]
    data_logging_r_mci_x = [r_mci[0]]
    data_logging_r_mci_y = [r_mci[1]]
    data_logging_r_mci_z = [r_mci[2]]
    data_logging_v_mci_x = [v_mci[0]]
    data_logging_v_mci_y = [v_mci[1]]
    data_logging_v_mci_z = [v_mci[2]]
    data_logging_thrust_mci_x = [thrust_vector_mci[0]]
    data_logging_thrust_mci_y = [thrust_vector_mci[1]]
    data_logging_thrust_mci_z = [thrust_vector_mci[2]]

    print('Starting run')

    i = 0
    while i < num_iterations and np.linalg.norm(r_mci) > planetary_body_config.body_radius:
        time += sim_dt
        r_mci, v_mci = orbital_dynamics.propagate_state_vector(sim_dt, r_mci, v_mci, thrust_vector_mci, mass)
        mass += -orbital_dynamics.mass_flow_rate(sim_configs.engine_isp, np.linalg.norm(thrust_vector_mci)) * sim_dt
        
        # TODO: bound thrust magnitudes
        if mass > sim_configs.dry_mass:
            thrust_vector_mci = mass * guidance.apollo_guidance_desired_acceleration_mci(500, time, 6, np.array([0.0, 0.0, 0.0]), v_mci, sim_configs.landing_site_r_mci, r_mci, np.array([0.0, 0.0, 0.0]))
        else:
            thrust_vector_mci = np.array([0.0, 0.0, 0.0])
        if np.linalg.norm(r_mci) - planetary_body_config.body_radius < 100:
            thrust_vector_mci = np.array([0.0, 0.0, 0.0])

        data_logging_time.append(time)
        data_logging_mass.append(mass)
        data_logging_r_mci_x.append(r_mci[0])
        data_logging_r_mci_y.append(r_mci[1])
        data_logging_r_mci_z.append(r_mci[2])
        data_logging_v_mci_x.append(v_mci[0])
        data_logging_v_mci_y.append(v_mci[1])
        data_logging_v_mci_z.append(v_mci[2])
        data_logging_thrust_mci_x.append(thrust_vector_mci[0])
        data_logging_thrust_mci_y.append(thrust_vector_mci[1])
        data_logging_thrust_mci_z.append(thrust_vector_mci[2])

        i += 1
        
    print('Run complete')
    output_data = [data_logging_time,
                   data_logging_mass,
                   data_logging_r_mci_x,
                   data_logging_r_mci_y,
                   data_logging_r_mci_z,
                   data_logging_v_mci_x,
                   data_logging_v_mci_y,
                   data_logging_v_mci_z,
                   data_logging_thrust_mci_x,
                   data_logging_thrust_mci_y,
                   data_logging_thrust_mci_z]
    
    sliced_output_data = []
    for data in output_data:
        sliced_output_data.append(data[::sim_configs.output_data_slice_factor])
    data_parser.logDataMCI(sliced_output_data)

    print('Data logged')




def run_sim_xyz():

    sim_dt = sim_configs.sim_dt
    sim_duration = sim_configs.sim_duration
    num_iterations = round(sim_duration/sim_dt)


    time = 0
    mass = sim_configs.wet_mass
    r_xyz = sim_configs.initial_r_xyz
    v_xyz = sim_configs.initial_v_xyz

    thrust_vector_xyz = np.array([0.0, 0.0, 0.0])

    data_logging_time = [time]
    data_logging_mass = [mass]
    data_logging_r_xyz_x = [r_xyz[0]]
    data_logging_r_xyz_y = [r_xyz[1]]
    data_logging_r_xyz_z = [r_xyz[2]]
    data_logging_v_xyz_x = [v_xyz[0]]
    data_logging_v_xyz_y = [v_xyz[1]]
    data_logging_v_xyz_z = [v_xyz[2]]
    data_logging_thrust_xyz_x = [thrust_vector_xyz[0]]
    data_logging_thrust_xyz_y = [thrust_vector_xyz[1]]
    data_logging_thrust_xyz_z = [thrust_vector_xyz[2]]

    print('Starting run')

    i = 0
    while i < num_iterations and r_xyz[0] > 0:
        time += sim_dt
        r_xyz, v_xyz = cartesian_dynamics.propagate_state_vector(sim_dt, r_xyz, v_xyz, thrust_vector_xyz, mass)
        mass += -cartesian_dynamics.mass_flow_rate(sim_configs.engine_isp, np.linalg.norm(thrust_vector_xyz)) * sim_dt
        
        # TODO: bound thrust magnitudes
        if mass > sim_configs.dry_mass:
            accel_vector_xyz = guidance.apollo_guidance_desired_acceleration_xyz(30, time, 6.0, np.array([0.0, 0.0, 0.0]), v_xyz, sim_configs.landing_site_r_xyz, r_xyz, np.array([0.0, 0.0, 0.0]))
            thrust_vector_xyz = mass * accel_vector_xyz
            # if np.linalg.norm(thrust_vector_xyz) > sim_configs.thrust_upper_bound:
            #     thrust_vector_xyz = thrust_vector_xyz / np.linalg.norm(thrust_vector_xyz) * sim_configs.thrust_upper_bound
            # elif np.linalg.norm(thrust_vector_xyz) < sim_configs.thrust_lower_bound:
            #     thrust_vector_xyz = thrust_vector_xyz / np.linalg.norm(thrust_vector_xyz) * sim_configs.thrust_lower_bound

        else:
            thrust_vector_xyz = np.array([0.0, 0.0, 0.0])
        if r_xyz[0] < 1:
            thrust_vector_xyz = np.array([0.0, 0.0, 0.0])
        

        # thrust_vector_xyz = np.array([0.0, 0.0, 0.0])


        data_logging_time.append(time)
        data_logging_mass.append(mass)
        data_logging_r_xyz_x.append(r_xyz[0])
        data_logging_r_xyz_y.append(r_xyz[1])
        data_logging_r_xyz_z.append(r_xyz[2])
        data_logging_v_xyz_x.append(v_xyz[0])
        data_logging_v_xyz_y.append(v_xyz[1])
        data_logging_v_xyz_z.append(v_xyz[2])
        data_logging_thrust_xyz_x.append(thrust_vector_xyz[0])
        data_logging_thrust_xyz_y.append(thrust_vector_xyz[1])
        data_logging_thrust_xyz_z.append(thrust_vector_xyz[2])

        i += 1
        
    print('Run complete')
    output_data = [data_logging_time,
                   data_logging_mass,
                   data_logging_r_xyz_x,
                   data_logging_r_xyz_y,
                   data_logging_r_xyz_z,
                   data_logging_v_xyz_x,
                   data_logging_v_xyz_y,
                   data_logging_v_xyz_z,
                   data_logging_thrust_xyz_x,
                   data_logging_thrust_xyz_y,
                   data_logging_thrust_xyz_z]
    
    sliced_output_data = []
    for data in output_data:
        sliced_output_data.append(data[::sim_configs.output_data_slice_factor])
    data_parser.logDataxyz(sliced_output_data)

    print('Data logged')