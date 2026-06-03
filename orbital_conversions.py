import numpy as np
import poliastro
import constants_and_helpers
import planetary_body_config


def MCI_to_MCMF(input_r_mci, input_v_mci):
    return input_r_mci, input_v_mci


def MCMF_to_MCI(input_r_mcmf, input_v_mcmf):
    return input_r_mcmf, input_v_mcmf

def NED_to_MCI(input_r_ned, input_v_ned):
    spherical_phi = input_r_ned[1] / planetary_body_config.body_radius
    spherical_theta = np.pi/2 - input_r_ned[0] / planetary_body_config.body_radius
    spherical_r = -input_r_ned[2] + planetary_body_config.body_radius
    output_r_mci = np.array([spherical_r*np.sin(spherical_theta)*np.cos(spherical_phi), spherical_r*np.sin(spherical_theta)*np.sin(spherical_phi), spherical_r*np.cos(spherical_theta)])
    
    spherical_phi_dot = input_v_ned[1] / planetary_body_config.body_radius
    spherical_theta_dot = -input_v_ned[0] / planetary_body_config.body_radius
    spherical_r_dot = -input_v_ned[2]
    output_v_mci_x = spherical_r_dot*np.sin(spherical_theta)*np.cos(spherical_phi) + spherical_r*(spherical_theta_dot*np.cos(spherical_theta)*np.cos(spherical_phi) - spherical_phi_dot*np.sin(spherical_phi)*np.sin(spherical_theta))
    output_v_mci_y = spherical_r_dot*np.sin(spherical_theta)*np.sin(spherical_phi) + spherical_r*(spherical_phi_dot*np.sin(spherical_theta)*np.cos(spherical_phi) + spherical_theta_dot*np.sin(spherical_phi_dot)*np.cos(spherical_theta))    
    output_v_mci_z = spherical_r_dot*np.cos(spherical_theta) - spherical_r*spherical_theta_dot*np.sin(spherical_theta)
    output_v_mci = np.array([output_v_mci_x, output_v_mci_y, output_v_mci_z])

    return output_r_mci, output_v_mci

def MCI_to_NED(input_r_mci):
    output_r_ned_d = -(np.linalg.norm(input_r_mci) - planetary_body_config.body_radius)
    output_r_ned_n = (np.pi/2 - np.arccos(input_r_mci[2] / np.linalg.norm(input_r_mci))) * planetary_body_config.body_radius
    output_r_ned_e = np.arccos(input_r_mci[0] / np.sqrt(input_r_mci[0]**2 + input_r_mci[1]**2))
    if input_r_mci[1] < 0:
        output_r_ned_e = -np.arccos(input_r_mci[0] / np.sqrt(input_r_mci[0]**2 + input_r_mci[1]**2))
    output_r_ned_e *= planetary_body_config.body_radius

    return np.array([output_r_ned_n, output_r_ned_e, output_r_ned_d])

def MCI_to_NED2(input_vector, input_r_mci):

    rotation_vector_1 = np.cross(np.array([1.0, 0.0, 0.0]), input_r_mci) / np.linalg.norm(np.cross(np.array([1.0, 0.0, 0.0]), input_r_mci))
    rotation_angle_1 = np.arccos(input_r_mci[0] / np.linalg.norm(input_r_mci))

    rotation_quaternion_1 = np.array([np.cos(rotation_angle_1/2), rotation_vector_1[0]*np.sin(rotation_angle_1/2), rotation_vector_1[1]*np.sin(rotation_angle_1/2), rotation_vector_1[2]*np.sin(rotation_angle_1/2)])
    rotated_vector_1 = constants_and_helpers.rotate_vector_by_quat(input_vector, rotation_quaternion_1)

    rotation_vector_2 = np.cross(np.array([0.0, 0.0, 1.0]), input_r_mci) / np.linalg.norm(np.cross(np.array([0.0, 0.0, 1.0]), input_r_mci))
    rotation_angle_2 = -np.pi/2

    rotation_quaternion_2 = np.array([np.cos(rotation_angle_2/2), rotation_vector_2[0]*np.sin(rotation_angle_2/2), rotation_vector_2[1]*np.sin(rotation_angle_2/2), rotation_vector_2[2]*np.sin(rotation_angle_2/2)])
    rotated_vector_2 = constants_and_helpers.rotate_vector_by_quat(rotated_vector_1, rotation_quaternion_2)


    return rotated_vector_2