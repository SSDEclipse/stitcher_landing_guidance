import numpy as np


g = 3.73

r_0 = np.array([100.0, 0.0, 0.0])
r_f = np.array([0.0, 0.0, 0.0])
v_0 = np.array([-45.0, 0.0, 0.0])
v_f_x = 0.0

t_f = 10

m_0 = 1000
T_max = 30000
T_min = 3000
Isp = 300

m_f = m_0 - T_max/(Isp*9.807) * t_f



c_0_x = -12.0 / t_f**4 * (1.0 / 6.0 * t_f**3 * (v_f_x - v_0[0]) - 1.0 / 2.0 * t_f**2 * (r_f[0] - v_0[0] * t_f - r_0[0])) + g



print()
print(c_0_x)
print(c_0_y)
print(c_0_z)
print()

print(m_0*(c_0_x**2 + c_0_y**2 + c_0_z**2)**0.5)
print(m_f*((c_0_x + c_1_x * t_f)**2 + (c_0_y + c_1_y * t_f)**2 + (c_0_z + c_1_z * t_f)**2)**0.5)

r_x_f = 0.5*(c_0_x - g)*t_f**2 + v_0[0]*t_f + r_0[0]
r_y_f = 0.5*c_0_y*t_f**2  + v_0[1]*t_f + r_0[1]
r_z_f = 0.5*c_0_z*t_f**2 + v_0[2]*t_f + r_0[2]

v_x_f = (c_0_x - g)*t_f + 0.5*c_1_x*t_f**2 + v_0[0]

print(r_x_f, r_y_f, r_z_f, v_x_f)