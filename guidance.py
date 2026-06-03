import numpy as np
import orbital_conversions
import orbital_dynamics
import cartesian_dynamics
import planetary_body_config


def apollo_guidance_desired_acceleration_mci(t_f, t, k_r, v_star_f, v_t, r_star_f, r_t, a_star_f):
    t_go = t_f - t
    if t_go < 0.1:
        t_go = 0.1
    gravity_vector = np.array([0.0, 0.0, -np.linalg.norm(orbital_dynamics.gravity_acceleration(r_t))])
    desired_acceleration_vector = 2/t_go * (1 - 1.0/3*k_r) * (v_star_f-v_t) + k_r/t_go**2 * (r_star_f-r_t-v_t*t_go) + 1.0/6 * (k_r-6)*a_star_f + 1.0/6 * (k_r-12)*gravity_vector

    return desired_acceleration_vector

def apollo_guidance_desired_acceleration_xyz(t_f, t, k_r, v_star_f, v_t, r_star_f, r_t, a_star_f):
    t_go = t_f - t
    gravity_vector = cartesian_dynamics.gravity_acceleration()
    desired_acceleration_vector = 2/t_go * (1 - 1.0/3*k_r) * (v_star_f-v_t) + k_r/t_go**2 * (r_star_f-r_t-v_t*t_go) + 1.0/6 * (k_r-6)*a_star_f + 1.0/6 * (k_r-12)*gravity_vector

    return desired_acceleration_vector
