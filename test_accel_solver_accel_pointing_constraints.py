import numpy as np


g = 3.73

t_f = 1.6

m_0 = 1000
T_max = 30000
T_min = 3000
Isp = 300

m_f = m_0 - T_max/(Isp*9.807) * t_f

r_0 = np.array([100.0, 0.0, 0.0])
r_f = np.array([0.0, 0.0, 0.0])
v_0 = np.array([-100.0, 0.0, 0.0])

a_0 = T_max/m_0 *  - v_0 / np.linalg.norm(v_0)

# m_f = m_0

c_0_x = a_0[0]
c_0_y = a_0[1]
c_0_z = a_0[2]

c_1_x = 6.0 / t_f**3 * (r_f[0] - r_0[0] - v_0[0]*t_f + 0.5*g*t_f**2 - 0.5*c_0_x*t_f**2)
c_1_y = 6.0 / t_f**3 * (r_f[1] - r_0[1] - v_0[1]*t_f + 0.5*c_0_y*t_f**2)
c_1_z = 6.0 / t_f**3 * (r_f[2] - r_0[2] - v_0[2]*t_f + 0.5*c_0_z*t_f**2)





print()
print(c_0_x, c_1_x)
print(c_0_y, c_1_y)
print(c_0_z, c_1_z)
print()

print(m_0*(c_0_x**2 + c_0_y**2 + c_0_z**2)**0.5)
print(m_f*((c_0_x + c_1_x * t_f/2)**2 + (c_0_y + c_1_y * t_f/2)**2 + (c_0_z + c_1_z * t_f/2)**2)**0.5)
print(m_f*((c_0_x + c_1_x * t_f)**2 + (c_0_y + c_1_y * t_f)**2 + (c_0_z + c_1_z * t_f)**2)**0.5)

r_x_f = 0.5*(c_0_x - g)*t_f**2 + 1/6.0 * c_1_x * t_f**3 + v_0[0]*t_f + r_0[0]
r_y_f = 0.5*c_0_y*t_f**2 + 1/6.0 * c_1_y * t_f**3 + v_0[1]*t_f + r_0[1]
r_z_f = 0.5*c_0_z*t_f**2 + 1/6.0 * c_1_z * t_f**3 + v_0[2]*t_f + r_0[2]

v_x_f = (c_0_x - g)*t_f + 0.5*c_1_x*t_f**2 + v_0[0]
v_y_f = c_0_y*t_f + 0.5*c_1_y*t_f**2 + v_0[1]
v_z_f = c_0_z*t_f + 0.5*c_1_z*t_f**2 + v_0[2]

print(r_x_f, r_y_f, r_z_f)
print(v_x_f, v_y_f, v_z_f)