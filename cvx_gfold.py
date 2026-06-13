import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R

# =============================================================================
# Define the time horizon and discretization
# =============================================================================
# T = 57.0  # Time horizon
# N = 100    # Number of discretization points
# dt = T / (N - 1)  # Time step

# r0 = np.array([[2400.0], [450.0], [-330.0]])
# v0 = np.array([[-10.0], [-40.0], [10.0]])
# rf = np.array([[0.0], [0.0], [0.0]])
# vf = np.array([[0.0], [0.0], [0.0]])

# m_wet = 2000.0  # mass, kg
# m_dry = 1700.0
# g = np.array([[3.7114], [0.0], [0.0]])
# T_max_thrust = 24000.0
# T_max = 0.8 * T_max_thrust  # max thrust force, N
# T_min = 0.2 * T_max_thrust  # min thrust force, N
# Isp = 1.0 / (9.80665 * 5 * 10**-4)
# alpha = 1.0 / (9.80665 * Isp)
# thrust_max_angle = 45.0 * np.pi / 180.0     # max vehicle tilt, rad
# glideslope_max_angle = 90.0 * np.pi / 180.0  # max glideslope angle



T = 20.0  # Time horizon
N = 50    # Number of discretization points
dt = T / (N - 1)  # Time step

r0 = np.array([[200.0], [0.0], [0.0]])
v0 = np.array([[-18.0], [0.0], [0.0]])
rf = np.array([[0.0], [0.0], [0.0]])
vf = np.array([[0.0], [0.0], [0.0]])

m_wet = 2000.0  # mass, kg
m_dry = 1000.0
g = np.array([[3.7278], [0.0], [0.0]])
T_max_thrust = 10000
T_max = 10000  # max thrust force, N
T_min = 3000  # min thrust force, N
Isp = 300.0
alpha = 1.0 / (9.80665 * Isp)
thrust_max_angle = 90.0 * np.pi / 180.0     # max vehicle tilt, rad
glideslope_max_angle = 90.0 * np.pi / 180.0  # max glideslope angle






# Scaling matrices
scaling_r = np.diag([max(1.0, abs(r0[0, 0])), max(1.0, abs(r0[1, 0])), max(1.0, abs(r0[2, 0]))])
scaling_v = np.diag([max(1.0, abs(v0[0, 0])), max(1.0, abs(v0[1, 0])), max(1.0, abs(v0[2, 0]))])
scaling_x = np.block([[scaling_r, np.zeros((3, 3))],
                      [np.zeros((3, 3)), scaling_v]])

scaling_z_v = (np.log(m_dry) + np.log(m_wet)) / 2.0
scaling_z_m = np.log(m_wet) - scaling_z_v
scaling_u_v = np.array([[0.5 * (T_max / m_wet + T_max / m_dry)], [0.0], [0.0]])
scaling_u_m = np.diag([T_max / m_dry - scaling_u_v[0, 0], 
                       T_max / m_dry * np.sin(thrust_max_angle), 
                       T_max / m_dry * np.sin(thrust_max_angle)])

scaling_sigma_0 = scaling_u_v[0, 0]
scaling_sigma_1 = scaling_u_m[0, 0]

# =============================================================================
# Define the CVXPY optimization variables
# =============================================================================
x_s = cp.Variable((6, N))       # x = [rx ry rz vx vy vz]
z_s = cp.Variable((1, N))       # z = ln mass
sigma_s = cp.Variable((1, N))   # thrust slack var
u_s = cp.Variable((3, N))       # Control input [ux; uy; uz]

# Define physical variables via scaling mappings
x = scaling_x @ x_s
z = scaling_z_m * z_s + scaling_z_v
# repmat(scaling_u_v, 1, N) becomes broadcasting addition in Python
u = scaling_u_m @ u_s + scaling_u_v  
sigma = scaling_sigma_1 * sigma_s + scaling_sigma_0

# =============================================================================
# Formulate Constraints and Objective
# =============================================================================
constraints = [
    x[0:3, 0:1] == r0, 
    x[3:6, 0:1] == v0, 
    z[0, 0] == np.log(m_wet),  # Initial states
    x[0:3, N-1:N] == rf, 
    x[3:6, N-1:N] == vf       # Final states
]

objective = 0.0

for k in range(N):
    # Add control effort to objective
    objective += sigma[0, k] * dt
    
    if k < N - 1:
        # Trapezoidal integration for dynamics
        constraints.append(x[3:6, k+1:k+2] == x[3:6, k:k+1] + dt/2.0 * ((u[:, k:k+1] - g) + (u[:, k+1:k+2] - g)))
        constraints.append(x[0:3, k+1:k+2] == x[0:3, k:k+1] + dt/2.0 * (x[3:6, k:k+1] + x[3:6, k+1:k+2]))
        # constraints.append(x[3:6, k+1:k+2] == x[3:6, k:k+1] + dt * ((u[:, k:k+1] - g)))
        # constraints.append(x[0:3, k+1:k+2] == x[0:3, k:k+1] + dt * (x[3:6, k:k+1]))
        constraints.append(z[0, k+1] == z[0, k] - (alpha * dt * sigma[0, k]))
    
    constraints.append(x[0, k] >= 0.0)

    # State mass bounds
    constraints.append(z[0, k] >= np.log(m_wet - alpha * T_max * (k) * dt))
    constraints.append(z[0, k] <= np.log(m_wet - alpha * T_min * (k) * dt))
    
    # Thrust Magnitude and Cone Constraints
    constraints.append(cp.norm(u[:, k]) <= sigma[0, k])
    constraints.append(u[0, k] >= np.cos(thrust_max_angle) * sigma[0, k])
    
    # Convex Lower and Upper Bounds for Thrust Slack Variable
    z0 = np.log(m_wet - alpha * T_max * (k) * dt)
    mu_1 = T_min * np.exp(-z0)
    mu_2 = T_max * np.exp(-z0)
    constraints.append(sigma[0, k] <= mu_2 * (1.0 - (z[0, k] - z0)))
    constraints.append(sigma[0, k] >= mu_1 * (1.0 - (z[0, k] - z0) + 0.5 * (z[0, k] - z0)**2))

# Solve the optimization problem using SCS
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve(solver=cp.SCS, verbose=True)

# =============================================================================
# Extract and Process Solution
# =============================================================================
u_opt = u.value
r_opt = x[0:3, :].value
v_opt = x[3:6, :].value
z_opt = z.value.flatten()
m_opt = np.exp(z_opt)

u_hat_opt = np.zeros((3, N))
u_df = np.zeros((3, N))

for i in range(N):
    T_vec = m_opt[i] * u_opt[:, i]
    u_df[:, i] = T_vec
    u_hat_opt[:, i] = T_vec / np.linalg.norm(T_vec)

u_hat_dot_opt = np.zeros((3, N))
for i in range(1, N):
    u_hat_dot_opt[:, i] = 1.0 / dt * (u_hat_opt[:, i] - u_hat_opt[:, i-1])

u_hat_double_dot_opt = np.zeros((3, N))
for i in range(1, N):
    u_hat_double_dot_opt[:, i] = 1.0 / dt * (u_hat_dot_opt[:, i] - u_hat_dot_opt[:, i-1])

n_hat_opt = np.zeros((3, N))
theta_opt = np.zeros(N)
rot_opt = np.zeros((3, 3 * N))
quat_opt = []

for i in range(N):
    n_hat_opt[:, i] = np.cross([1, 0, 0], u_hat_opt[:, i])
    theta_opt[i] = np.arccos(u_hat_opt[0, i])
    
    # Constructing Quaternion elements
    q_w = np.cos(theta_opt[i] / 2.0)
    q_vec = n_hat_opt[:, i] * np.sin(theta_opt[i] / 2.0)
    # SciPy uses scalar-last quaternion format: [x, y, z, w]
    quat_opt.append(R.from_quat([q_vec[0], q_vec[1], q_vec[2], q_w]))

for i in range(N):
    n = n_hat_opt[:, i]
    n_hat_opt_x = np.array([[0, -n[2], n[1]],
                            [n[2], 0, -n[0]],
                            [-n[1], n[0], 0]])
    
    # Rodrigues' formula implementation
    rot_mat = np.eye(3) + np.sin(theta_opt[i]) * n_hat_opt_x + (1.0 - np.cos(theta_opt[i])) * np.linalg.matrix_power(n_hat_opt_x, 2)

    
    rot_opt[:, 3*i : 3*(i+1)] = rot_mat

w_opt = np.zeros((3, N))
for i in range(N):
    w_opt[:, i] = rot_opt[:, 3*i : 3*(i+1)].T @ u_hat_dot_opt[:, i]

# =============================================================================
# Plotting Trajectories
# =============================================================================
# plt.rcParams.update({'text.usetex': True, 'font.size': 14}) # Match LaTeX styling

# Calculate speed (velocity magnitude) for color mapping
speed = np.linalg.norm(v_opt, axis=0)
t = np.linspace(0, T, N)

# =============================================================================
# Figure 1: Interactive 3D Trajectory with Thrust Quivers
# =============================================================================
fig1 = go.Figure()

# Plot the 3D trajectory line using a single scatter trace that maps color to the line segments
fig1.add_trace(go.Scatter3d(
    x=r_opt[2, :],  # -zz
    y=r_opt[1, :],   # yy
    z=r_opt[0, :],   # xx
    mode='lines',
    marker=dict(
        size=4,
        color=speed,
        colorscale='Turbo',
        showscale=False  # Hide marker scale to avoid duplicating the main line colorbar
    ),
    line=dict(
        color=speed,      # This applies the speed gradient directly to the trajectory line
        colorscale='Turbo',
        width=6,
        colorbar=dict(title="Speed (m/s)", x=0.85)
    ),
    name='Trajectory'
))

# Add thrust vectors as uniform red lines (no color gradient applied here)
scale_factor = 50  # Adjust this to scale the thrust line lengths visually
for i in range(N):
    fig1.add_trace(go.Scatter3d(
        x=[r_opt[2, i], r_opt[2, i] + scale_factor * u_opt[2, i]],
        y=[r_opt[1, i], r_opt[1, i] + scale_factor * u_opt[1, i]],
        z=[r_opt[0, i], r_opt[0, i] + scale_factor * u_opt[0, i]],
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    ))

fig1.update_layout(
    title='Trajectory',
    scene=dict(
        xaxis_title='Z',
        yaxis_title='Y',
        zaxis_title='X',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0))
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
fig1.show()

# =============================================================================
# Figure 2: Position and Velocity Time Series Subplots
# =============================================================================
fig2 = make_subplots(rows=2, cols=1, subplot_titles=('Optimal Position Trajectory', 'Optimal Velocity Trajectory'))

# Position Subplot
fig2.add_trace(go.Scatter(x=t, y=r_opt[0, :], name='x', line=dict(color='red')), row=1, col=1)
fig2.add_trace(go.Scatter(x=t, y=r_opt[1, :], name='y', line=dict(color='green')), row=1, col=1)
fig2.add_trace(go.Scatter(x=t, y=r_opt[2, :], name='z', line=dict(color='blue')), row=1, col=1)

# Velocity Subplot
fig2.add_trace(go.Scatter(x=t, y=v_opt[0, :], name='vx', line=dict(color='red')), row=2, col=1)
fig2.add_trace(go.Scatter(x=t, y=v_opt[1, :], name='vy', line=dict(color='green')), row=2, col=1)
fig2.add_trace(go.Scatter(x=t, y=v_opt[2, :], name='vz', line=dict(color='blue')), row=2, col=1)

fig2.update_xaxes(title_text="Time (s)", row=2, col=1)
fig2.update_yaxes(title_text="Position (m)", row=1, col=1)
fig2.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
fig2.update_layout(height=700, title_text="Kinematics Time Series", showlegend=True)
fig2.show()

# =============================================================================
# Figure 3: Optimal Control Inputs
# =============================================================================
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=u_opt[0, :], name='ux', line=dict(color='red')))
fig3.add_trace(go.Scatter(x=t, y=u_opt[1, :], name='uy', line=dict(color='green')))
fig3.add_trace(go.Scatter(x=t, y=u_opt[2, :], name='uz', line=dict(color='blue')))

fig3.update_layout(
    title='Optimal Control Input',
    xaxis_title='Time (s)',
    yaxis_title='Control Input (N/kg)',
    showlegend=True
)
fig3.show()

# =============================================================================
# Figure 4 & 5: Rocket Mass & Thrust Magnitude Profile
# =============================================================================
fig4_5 = make_subplots(rows=2, cols=1, subplot_titles=('Mass History', 'Thrust Magnitude Profile'))

# Mass plot
fig4_5.add_trace(go.Scatter(x=t, y=m_opt, name='Mass', line=dict(color='purple')), row=1, col=1)

# Thrust Magnitude plot
thrust_mag = (1.0 / T_max_thrust) * np.sqrt(np.sum(u_opt**2, axis=0)) * m_opt
fig4_5.add_trace(go.Scatter(x=t, y=thrust_mag, name='Thrust Profile', line=dict(color='orange')), row=2, col=1)

fig4_5.update_xaxes(title_text="Time (s)", row=2, col=1)
fig4_5.update_yaxes(title_text="Mass (kg)", row=1, col=1)
fig4_5.update_yaxes(title_text="Normalized Thrust", row=2, col=1)
fig4_5.update_layout(height=600, title_text="Vehicle Status Over Time", showlegend=False)
fig4_5.show()