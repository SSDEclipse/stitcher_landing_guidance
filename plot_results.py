import numpy as np
import plotting_functions
import sim_configs
import data_parser

output_data = data_parser.OutputDataXYZ('outputs/standalone/data.csv')


print(output_data.time)

plotting_functions.plot_3d_data(output_data.time, -output_data.r_xyz[:, 2], output_data.r_xyz[:, 1], output_data.r_xyz[:, 0])

plotting_functions.plot_2d_data([output_data.time], [output_data.thrust_mag], [''], 'Thrust magnitude vs time', 'Time (s)', 'Thrust (N)')

plotting_functions.plot_2d_data([output_data.time], [output_data.thrust_mag / sim_configs.thrust_max], [''], 'Throttle vs time', 'Time (s)', 'Throttle')

plotting_functions.plot_2d_data([output_data.time, output_data.time, output_data.time], [output_data.thrust_vector_xyz[:, 0], output_data.thrust_vector_xyz[:, 1], output_data.thrust_vector_xyz[:, 2]], ['tx', 'ty', 'tz'], 'Thrust vs time', 'Time (s)', 'Thrust (N)')


plotting_functions.plot_2d_data([output_data.time], [output_data.v_xyz_mag], [''], 'Velocity magnitude vs time', 'Time (s)', 'Velocity (m/s)')

plotting_functions.plot_2d_data([output_data.time, output_data.time, output_data.time], [output_data.v_xyz[:, 0], output_data.v_xyz[:, 1], output_data.v_xyz[:, 2]], ['vx', 'vy', 'vz'], 'Velocity vs time', 'Time (s)', 'Velocity (m/s)')

plotting_functions.plot_2d_data([output_data.time, output_data.time, output_data.time], [output_data.r_xyz[:, 0], output_data.r_xyz[:, 1], output_data.r_xyz[:, 2]], ['rx', 'ry', 'rz'], 'Position vs time', 'Time (s)', 'Position (m)')