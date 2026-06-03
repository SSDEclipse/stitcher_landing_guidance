import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. GENERATE SYNTHETIC 3D DATA
# ==========================================
import data_parser

output_data = data_parser.OutputData('outputs/standalone/data.csv')
sliceFactor = 1
t = output_data.time
pos_array = output_data.r_ned/1000.0
thrust_array = output_data.thrust_vector_ned
print(thrust_array)
x, y, z = pos_array[:, 0], pos_array[:, 1], -pos_array[:, 2]
thrust_x, thrust_y, thrust_z = -thrust_array[:, 0], -thrust_array[:, 1], -thrust_array[:, 2] 

ROCKET_LENGTH = 1.0
THRUST_SCALE = 0.0001




nose_coords = []
tail_coords = []
thrust_end_coords = []

for i in range(len(t)):
    cx, cy, cz = x[i], y[i], z[i]
    tx, ty, tz = thrust_x[i], thrust_y[i], thrust_z[i]
    
    raw_thrust = np.array([tx, ty, tz])
    t_norm = np.linalg.norm(raw_thrust)
    
    if t_norm > 0:
        thrust_dir = raw_thrust / t_norm
        heading_dir = -thrust_dir
    else:
        heading_dir = np.array([0, 0, 1])
        thrust_dir = np.array([0, 0, -1])
        
    nose = np.array([cx, cy, cz]) + heading_dir * (ROCKET_LENGTH / 2.0)
    tail = np.array([cx, cy, cz]) - heading_dir * (ROCKET_LENGTH / 2.0)
    thrust_end = tail + (thrust_dir * t_norm * THRUST_SCALE)
    
    nose_coords.append(nose)
    tail_coords.append(tail)
    thrust_end_coords.append(thrust_end)

# ==========================================
# 2. INITIALIZE BASE PLOTLY FIGURES
# ==========================================
fig = go.Figure()

# Trace 0: Static Trajectory Trace
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, mode='lines',
    line=dict(color='#4f46e5', width=3, dash='dash'),
    name='Full Trajectory', opacity=0.4
))

# Trace 1: Rocket Body
fig.add_trace(go.Scatter3d(
    x=[tail_coords[0][0], nose_coords[0][0]],
    y=[tail_coords[0][1], nose_coords[0][1]],
    z=[tail_coords[0][2], nose_coords[0][2]],
    mode='lines', line=dict(color='#e2e8f0', width=8), name='Rocket Body'
))

# Trace 2: Thrust Vector
fig.add_trace(go.Scatter3d(
    x=[tail_coords[0][0], thrust_end_coords[0][0]],
    y=[tail_coords[0][1], thrust_end_coords[0][1]],
    z=[tail_coords[0][2], thrust_end_coords[0][2]],
    mode='lines', line=dict(color='#f97316', width=5), name='Thrust Vector'
))

# Trace 3: Center of Mass Marker
fig.add_trace(go.Scatter3d(
    x=[x[0]], y=[y[0]], z=[z[0]],
    mode='markers', marker=dict(color='#ef4444', size=5), name='Center of Mass'
))

# ==========================================
# 3. BUILD ANIMATION FRAME DICTIONARIES
# ==========================================
frames = []
for i in range(len(t)):
    frames.append(go.Frame(
        data=[
            # EVERY frame must explicitly declare all matching trace allocations 
            # in identical structural array orders for the JS engine map to attach
            go.Scatter3d(x=x, y=y, z=z),
            go.Scatter3d(x=[tail_coords[i][0], nose_coords[i][0]], y=[tail_coords[i][1], nose_coords[i][1]], z=[tail_coords[i][2], nose_coords[i][2]]),
            go.Scatter3d(x=[tail_coords[i][0], thrust_end_coords[i][0]], y=[tail_coords[i][1], thrust_end_coords[i][1]], z=[tail_coords[i][2], thrust_end_coords[i][2]]),
            go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]])
        ],
        name=f"frame_{i}",
        layout=go.Layout(
            annotations=[dict(
                text=f"MET: {t[i]:05.2f} s",
                x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False,
                font=dict(family="Courier New, monospace", size=18, color="#38bdf8")
            )]
        )
    ))

fig.frames = frames

# ==========================================
# 4. CONFIGURING THE SLIDER STEPS
# ==========================================
slider_steps = []
for i in range(len(t)):
    step = dict(
        method="animate",
        # FIX: Flip redraw to True so 3D objects process visual updates on click/scrub
        args=[[f"frame_{i}"], dict(
            mode="immediate",
            frame=dict(duration=0, redraw=True),
            transition=dict(duration=0)
        )],
        label=f"{t[i]:.1f}s"
    )
    slider_steps.append(step)

sliders_config = [dict(
    active=0,
    currentvalue=dict(
        visible=True, 
        prefix="Timeline Point: ", 
        xanchor="right", 
        font=dict(size=14, color="#e2e8f0")
    ),
    pad={"b": 10, "t": 60},
    len=0.9,
    x=0.1, y=0,
    steps=slider_steps
)]

# ==========================================
# 5. DASHBOARD LAYOUT & UI BUTTONS
# ==========================================
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='#0d1117',
    scene_bgcolor='#0d1117',
    scene=dict(
        xaxis=dict(title='X Position (m)', range=[min(x)-10, max(x)+10]),
        yaxis=dict(title='Y Position (m)', range=[min(y)-10, max(y)+10]),
        zaxis=dict(title='Altitude Z (m)', range=[0, max(z)+10]),
        aspectmode='cube'
    ),
    title="3D Rocket Telemetry Animation Dashboard",
    margin=dict(l=0, r=0, b=40, t=50),
    sliders=sliders_config,
    
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(
                label="Play",
                method="animate",
                # FIX: Flip redraw to True here to enable the playback cycle loops
                args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, transition=dict(duration=0))]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
            )
        ],
        direction="left",
        pad={"r": 10, "t": 70},
        showactive=False,
        x=0.08, y=0,
        xanchor="right", yanchor="top"
    )]
)

# Initial global layout annotation state anchor
fig.update_layout(annotations=[dict(
    text=f"MET: {t[0]:05.2f} s",
    x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False,
    font=dict(family="Courier New, monospace", size=18, color="#38bdf8")
)])

fig.show()