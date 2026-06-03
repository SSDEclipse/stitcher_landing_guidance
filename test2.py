import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button  # Required for UI interaction

import data_parser

# ==========================================
# 1. GENERATE SYNTHETIC 3D DATA (For testing)
# ==========================================

output_data = data_parser.OutputData('outputs/standalone/data.csv')
sliceFactor = 1
thrust_scale_factor = 1000
t = output_data.time
pos_array = output_data.r_mci[::sliceFactor]/1000.0
thrust_array = output_data.thrust_vector_mci[::sliceFactor]/thrust_scale_factor

x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]
thrust_x, thrust_y, thrust_z = thrust_array[:, 0], thrust_array[:, 1], thrust_array[:, 2] 

# ==========================================
# 2. ANIMATION & UI SETUP
# ==========================================
fig = plt.figure(figsize=(10, 8))
# Leave a bit of room at the bottom for the UI button
ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], projection='3d')

fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Plot the static full trajectory line
ax.plot(x, y, z, color="#6863c2", linestyle='--', alpha=0.4, label='Trajectory')

# Initialize the dynamic 3D elements
rocket_body_line, = ax.plot([], [], [], color='#e2e8f0', linewidth=5, solid_capstyle='round', label='Rocket Body')
thrust_vector_line, = ax.plot([], [], [], color='#f97316', linewidth=3, label='Thrust Vector')
rocket_position_dot, = ax.plot([], [], [], 'ro', markersize=6)

time_text = fig.text(
    0.12, 0.85, '', 
    color='#38bdf8',         # Cyan readout color
    fontsize=14, 
    fontfamily='monospace',   # Prevents numbers from stuttering horizontally
    weight='bold'
)

# Graph styling
ax.set_xlim(min(x) - 50, max(x) + 50)
ax.set_ylim(min(y) - 50, max(y) + 50)
ax.set_zlim(min(z) - 50, max(z) + 50)
ax.set_xlabel('X Position (m)', color='#e2e8f0')
ax.set_ylabel('Y Position (m)', color='#e2e8f0')
ax.set_zlabel('Altitude Z (m)', color='#e2e8f0')
ax.set_title('3D Rocket Trajectory (Click Button to Pause)', color='#e2e8f0', fontsize=14)

ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#0d1117')
ax.yaxis.pane.set_edgecolor('#0d1117')
ax.zaxis.pane.set_edgecolor('#0d1117')
ax.tick_params(colors='#e2e8f0')
ax.grid(True, linestyle=':', alpha=0.2)
# ax.legend(loc='upper left', framealpha=0.2, facecolor='#e2e8f0', edgecolor='none')

ROCKET_LENGTH = 50.0  
THRUST_SCALE = 0.4  

# ==========================================
# 3. INTERACTIVE CONTROLS (PLAY/PAUSE)
# ==========================================
# Track animation state
is_running = True

# Define an axis area for the button [left, bottom, width, height]
ax_button = fig.add_axes([0.45, 0.03, 0.1, 0.05])
btn_pause = Button(
    ax_button, 
    label='Pause', 
    color='#1f2937',      # Dark grey button background
    hovercolor='#374151'  # Lighter grey on hover
)
btn_pause.label.set_color('#e2e8f0')  # Off-white text

def toggle_animation(event):
    global is_running
    if is_running:
        ani.pause()
        btn_pause.label.set_text('Play')
    else:
        ani.resume()
        btn_pause.label.set_text('Pause')
    is_running = not is_running
    fig.canvas.draw_idle()  # Force the UI text layer to update immediately

# Link the button click event to our function
btn_pause.on_clicked(toggle_animation)

# ==========================================
# 4. 3D ANIMATION LOOP FUNCTION
# ==========================================
def update(frame):
    # 1. Fetch current timestamp from your time array
    current_time = t[frame]
    
    cx, cy, cz = x[frame], y[frame], z[frame]
    tx, ty, tz = thrust_x[frame], thrust_y[frame], thrust_z[frame]
    
    raw_thrust = np.array([tx, ty, tz])
    thrust_norm = np.linalg.norm(raw_thrust)
    
    if thrust_norm > 0:
        thrust_dir = raw_thrust / thrust_norm
        heading_dir = -thrust_dir 
    else:
        heading_dir = np.array([0, 0, 1])

    half_len = ROCKET_LENGTH / 2.0
    nose = np.array([cx, cy, cz]) + heading_dir * half_len
    tail = np.array([cx, cy, cz]) - heading_dir * half_len
    thrust_end = tail + (thrust_dir * thrust_norm * THRUST_SCALE)
    
    # Update position and vector lines
    rocket_body_line.set_data([tail[0], nose[0]], [tail[1], nose[1]])
    rocket_body_line.set_3d_properties([tail[2], nose[2]])
    
    thrust_vector_line.set_data([tail[0], thrust_end[0]], [tail[1], thrust_end[1]])
    thrust_vector_line.set_3d_properties([tail[2], thrust_end[2]])
    
    rocket_position_dot.set_data([cx], [cy])
    rocket_position_dot.set_3d_properties([cz])
    
    # 2. Update the dynamic clock readout text (MET = Mission Elapsed Time)
    time_text.set_text(f"MET: {current_time:05.2f} s") 
        
    # 3. Append time_text to your returned artist tuples
    return rocket_body_line, thrust_vector_line, rocket_position_dot, time_text

# Create the 3D animation object
ani = animation.FuncAnimation(
    fig, update, frames=len(t), interval=50, blit=False
)

plt.show()