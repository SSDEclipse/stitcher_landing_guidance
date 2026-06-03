import numpy as np
import planetary_body_config
import orbital_conversions

output_data_slice_factor = 10

sim_dt = 0.01
sim_duration = 0.1*60*60

# initial_r_mci = np.array([1838000.0, 0.0, 0.0])
# initial_v_mci = np.array([0.0, 1156.0, 1156.0])


# initial_r_ned = np.array([10000.0, -00000.0, -20000.0])
# initial_v_ned = np.array([0.0, 0.0, 0.0])
# initial_r_mci, initial_v_mci = orbital_conversions.NED_to_MCI(initial_r_ned, initial_v_ned)




# landing_site_r_mci = np.array([planetary_body_config.body_radius, 0.0, 0.0])
# mci_2_landing_site_quat = np.array([1.0, 0.0, 0.0, 0.0])

wet_mass = 1905
dry_mass = 1000
engine_isp = 225
thrust_max = 3.1*1000*6
thrust_upper_bound = 0.8*thrust_max
thrust_lower_bound = 0.3*thrust_max

# gfold paper initial conditions
initial_r_xyz = np.array([1500.0, 0.0, 2000.0])
initial_v_xyz = np.array([-75.0, 0.0, 100.0])
# initial_r_xyz = np.array([1500.0, 0.0, 2000.0])
# initial_v_xyz = np.array([-75.0, 0.0, 0.0])
landing_site_r_xyz = np.array([0.0, 0.0, 0.0])