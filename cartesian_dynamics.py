import numpy as np
import constants_and_helpers
import planetary_body_config


def gravity_acceleration():
    return -constants_and_helpers.gravitational_constant*planetary_body_config.body_mass/(np.linalg.norm(planetary_body_config.body_radius)**2) * np.array([1.0, 0.0, 0.0])

def propagate_state_vector(input_dt, input_r_xyz, input_v_xyz, input_thrust_xyz, input_mass):
    accel_vector_xyz = gravity_acceleration() + input_thrust_xyz/input_mass
    output_v_xyz = input_v_xyz + accel_vector_xyz * input_dt
    output_r_xyz = input_r_xyz + output_v_xyz * input_dt
    return output_r_xyz, output_v_xyz


def mass_flow_rate(isp, thrust):
    return thrust/(isp*constants_and_helpers.g0)