import numpy as np
import constants_and_helpers
import planetary_body_config


def gravity_acceleration(input_r):
    return -input_r * constants_and_helpers.gravitational_constant*planetary_body_config.body_mass/(np.linalg.norm(input_r)**3)

def propagate_state_vector(input_dt, input_r_mci, input_v_mci, input_thrust_mci, input_mass):
    accel_vector_mci = gravity_acceleration(input_r_mci) + input_thrust_mci/input_mass
    output_v_mci = input_v_mci + accel_vector_mci * input_dt
    output_r_mci = input_r_mci + output_v_mci * input_dt
    return output_r_mci, output_v_mci


def mass_flow_rate(isp, thrust):
    return thrust/(isp*constants_and_helpers.g0)