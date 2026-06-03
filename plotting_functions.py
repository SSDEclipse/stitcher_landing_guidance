import data_parser
import sim_configs
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

output_data = data_parser.OutputDataXYZ('outputs/standalone/data.csv')


default_num_frames = len(output_data.time)

aniFPS = 10
timeBetweenFrames = 1.0/aniFPS*1000
sliceFactor = 10


def animate_orbit():


    frame_duration = timeBetweenFrames

    def create_sphere(radius, n_points=10):
        """Generates X, Y, Z coordinates for a sphere."""
        phi = np.linspace(0, np.pi, n_points)
        theta = np.linspace(0, 2 * np.pi, n_points)
        phi, theta = np.meshgrid(phi, theta)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return x, y, z

    def animate_orbit_with_slider(time, pos_array, body_radius=1738):
        x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]
        
        fig = go.Figure()

        # Trace 0: The Central Body
        sx, sy, sz = create_sphere(body_radius)
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz, 
            colorscale='Greys', 
            showscale=False, 
            opacity=0.8,
            name='Central Body',
            hoverinfo='skip'
        ))

        # Trace 1: Static Full Path
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='white', width=1),
            opacity=0.1,
            name='Full Orbit'
        ))

        # Trace 2: Dynamic Traced Path
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Traced Path'
        ))

        # Trace 3: Current Position Marker
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(color='red', size=5),
            name='Vehicle'
        ))

        # 1. Define Frames and Slider Steps
        frames = []
        slider_steps = []
        
        # Using a step of 2 for performance, same as before
        for i in range(0, len(time), 2):
            frame_name = f"step_{i}"
            
            # Create the frame
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=x[:i+1], y=y[:i+1], z=z[:i+1]), # Update trace 2
                    go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]])     # Update trace 3
                ],
                traces=[2, 3],
                name=frame_name
            ))
            
            # Create the slider step for this frame
            slider_step = {
                "args": [
                    [frame_name],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
                ],
                "label": f"{time[i]:.1f}s", # The text shown on the slider
                "method": "animate"
            }
            slider_steps.append(slider_step)

        fig.frames = frames

        # 2. Add Slider and Play/Pause to Layout
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
                aspectmode='data'
            ),
            updatemxyzs=[{
                "buttons": [
                            {
                                # The 'duration' here sets the speed of the Play button
                                "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}],
                                "label": "Play", "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                                "label": "Pause", "method": "animate"
                            }
                        ],
                "type": "buttons", "showactive": False, "x": 0, "y": 0, "xanchor": "right", "yanchor": "top"
            }],
            sliders=[{
                "steps": slider_steps,
                "x": 0.1, 
                "len": 0.9, 
                "y": 0,
                "currentvalue": {
                    "prefix": f"Time: ", 
                    "font": {"size": 16},
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps
            }]
        )

        fig.show()

    t_vec = output_data.time[::sliceFactor]

    pos_vec = output_data.r_mci[::sliceFactor]/1000.0

    animate_orbit_with_slider(t_vec, pos_vec)



def plot_3d_data(data_t, data_x, data_y, data_z):

    fig = go.Figure()

    # Add the 3D path trace
    fig.add_trace(go.Scatter3d(
        x=data_x,
        y=data_y,
        z=data_z,
        mode='lines+markers',             # Draws the continuous line AND color-mapped points
        
        # Line styling (the physical wireframe)
        line=dict(
            color='rgba(100, 100, 100, 0.4)', # Semi-transparent grey line connecting everything
            width=4
        ),
        
        # Marker styling (handles the time-based color gradient)
        marker=dict(
            size=4,
            color=data_t,          # Maps colors directly to your time array
            colorscale='Viridis',         # Clean time-progression color scheme
            colorbar=dict(title="Time"),  # Adds the legend bar on the right
            opacity=0.9
        ),
        
        # Hover text configurations
        customdata=data_t,
        hovertemplate=(
            "<b>Time:</b> %{customdata:.1f}<br>" +
            "<b>X:</b> %{x:.2f}<br>" +
            "<b>Y:</b> %{y:.2f}<br>" +
            "<b>Z:</b> %{z:.2f}<extra></extra>" # <extra></extra> hides the default secondary box
        )
    ))

    # --- 3. Scene Layout Adjustments ---
    fig.update_layout(
        title="3D Time-Series Trajectory Plot",
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            bgcolor="rgb(250, 250, 250)"
        ),
    )

    # Display the final plot
    fig.show()


def plot_2d_data(x_data, y_data, labels, title, xaxis_label, yaxis_label):
    """Plots multiple time series where the x-axis is numeric elapsed time.

        Parameters:
        -----------
        x_data : list of np.ndarray
            List containing numpy arrays for the x-axis (elapsed time).
        y_data : list of np.ndarray
            List containing numpy arrays for the y-axis (values).
        labels : list of str
            List of strings for the legend labels.
        title : str
            The main title of the plot.
        xaxis_label : str
            The label for the x-axis.
        yaxis_label : str
            The label for the y-axis.
        """
    fig = go.Figure()

    # Loop through and add each time series trace
    for x, y, label in zip(x_data, y_data, labels):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                connectgaps=True,
            )
        )

    # Update layout using the explicit string arguments
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_label,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        xaxis=dict(type="linear"),
    )

    fig.show()