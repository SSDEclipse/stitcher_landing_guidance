import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. GENERATE SYNTHETIC DATA (For testing)
# ==========================================
# Let's simulate a pitch-over maneuver (a gravity turn)
t = np.linspace(0, 20, 200)

# Rocket position (X, Y) over time
x = 50 * t + 2 * t**2
y = 100 * t - 3 * t**2 + 0.1 * t**3

# Rocket attitude angle theta (in radians) - where the nose points
# It starts pointing straight up (pi/2) and slowly pitches over
theta = np.pi/2 - (t / 20) * (np.pi/3)

# Thrust magnitude and vector direction
# Let's assume thrust is mostly aligned with the body, but has some gimbal adjustments
thrust_mag = 150 - 2 * t  # Burning fuel, getting lighter/throttling down
gimbal_angle = 0.05 * np.sin(2 * t)  # Small steering adjustments
thrust_angle = theta + np.pi + gimbal_angle  # Points opposite to direction of push

# Thrust vector components
thrust_x = thrust_mag * np.cos(thrust_angle)
thrust_y = thrust_mag * np.sin(thrust_angle)


# ==========================================
# 2. ANIMATION SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('#0d1117')  # Dark space-like background
fig.patch.set_facecolor('#0d1117')

# Draw the continuous trajectory path line
ax.plot(x, y, color='#4f46e5', linestyle='--', alpha=0.5, label='Trajectory')

# Initialize the components we want to animate
rocket_body_line, = ax.plot([], [], color='#e2e8f0', linewidth=4, solid_capstyle='round', label='Rocket Body')
thrust_vector_line, = ax.plot([], [], color='#f97316', linewidth=3, label='Thrust Vector')
rocket_position_dot, = ax.plot([], [], 'ro', markersize=6)

# Graph styling
ax.set_xlim(min(x) - 100, max(x) + 100)
ax.set_ylim(min(y) - 100, max(y) + 100)
ax.set_xlabel('X Position (m)', color='#e2e8f0')
ax.set_ylabel('Y Position (m)', color='#e2e8f0')
ax.set_title('Rocket Trajectory & Thrust Vector Animation', color='#e2e8f0', fontsize=14)
ax.tick_params(colors='#e2e8f0')
ax.grid(True, linestyle=':', alpha=0.3)
ax.legend(loc='upper left', framealpha=0.2, facecolor='#e2e8f0', edgecolor='none')

# Physical scale of the visualization assets
ROCKET_LENGTH = 40  
THRUST_SCALE = 0.5  # Scale down thrust magnitude so it fits nicely on screen

# ==========================================
# 3. ANIMATION LOOP FUNCTION
# ==========================================
def update(frame):
    # Get current state data
    curr_x = x[frame]
    curr_y = y[frame]
    curr_theta = theta[frame]
    curr_tx = thrust_x[frame] * THRUST_SCALE
    curr_ty = thrust_y[frame] * THRUST_SCALE
    
    # 1. Update Rocket Body (represented as a line segment)
    # Center of mass is at (curr_x, curr_y)
    # Nose point
    nose_x = curr_x + (ROCKET_LENGTH / 2) * np.cos(curr_theta)
    nose_y = curr_y + (ROCKET_LENGTH / 2) * np.sin(curr_theta)
    # Tail point
    tail_x = curr_x - (ROCKET_LENGTH / 2) * np.cos(curr_theta)
    tail_y = curr_y - (ROCKET_LENGTH / 2) * np.sin(curr_theta)
    
    rocket_body_line.set_data([tail_x, nose_x], [tail_y, nose_y])
    
    # 2. Update Thrust Vector (pointing outward from the tail)
    thrust_end_x = tail_x + curr_tx
    thrust_end_y = tail_y + curr_ty
    thrust_vector_line.set_data([tail_x, thrust_end_x], [tail_y, thrust_end_y])
    
    # 3. Update center of mass dot tracker
    rocket_position_dot.set_data([curr_x], [curr_y])
    
    return rocket_body_line, thrust_vector_line, rocket_position_dot

# Create the animation object
ani = animation.FuncAnimation(
    fig, update, frames=len(t), interval=50, blit=True
)

# To view it locally, uncomment the line below:
plt.show()

# Or to save it as an MP4 video file:
# ani.save('rocket_trajectory.mp4', writer='ffmpeg', fps=30)