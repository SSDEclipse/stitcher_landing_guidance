import numpy as np
import cartesian_dynamics

# moon
# body_mass = 7.346*10**22

# body_radius = 1737.4*10**3

# body_omega_rad_s = 2*np.pi/(27.321661*24*60*60)


# mars
body_mass = 6.4171*10**23

body_radius = 3389.5*10**3

body_omega_rad_s = 2*np.pi/(24.622*60*60)

body_surface_gravity = np.linalg.norm(cartesian_dynamics.gravity_acceleration())