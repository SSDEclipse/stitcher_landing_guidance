import numpy as np
import plotly.graph_objects as go


def generate_cylinder_mesh(bx, by, bz, dx, dy, dz, radius, length, n_sides=12):
    """Generates 3D coordinates and triangulation indices for a single rocket cylinder."""
    w = np.array([dx, dy, dz], dtype=float)
    up = np.array([0, 0, 1]) if abs(w[2]) < 0.9 else np.array([1, 0, 0])

    u = np.cross(up, w)
    u_norm = np.linalg.norm(u)
    u = u / u_norm if u_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    v = np.cross(w, u)

    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    base_pts = [
        np.array([bx, by, bz]) + radius * (np.cos(t) * u + np.sin(t) * v)
        for t in theta
    ]
    top_pts = [pt + length * w for pt in base_pts]
    pts = base_pts + top_pts

    cyl_x = [p[0] for p in pts]
    cyl_y = [p[1] for p in pts]
    cyl_z = [p[2] for p in pts]

    i_list, j_list, k_list = [], [], []
    for s in range(n_sides):
        next_s = (s + 1) % n_sides
        b1, b2 = s, next_s
        t1, t2 = n_sides + s, n_sides + next_s

        i_list.extend([b1, t1])
        j_list.extend([b2, t2])
        k_list.extend([t1, b2])

    return cyl_x, cyl_y, cyl_z, i_list, j_list, k_list


def plot_3d_data_with_rocket(
    data_t,
    data_x,
    data_y,
    data_z,
    acc_t,
    acc_x,
    acc_y,
    acc_z,
    rocket_length=40.0,
    rocket_radius=4.5,
    thrust_scale=3,
    polygon_coords=None,
    fps=10,
):
    frame_duration_ms = int(1000 / fps)  # 100 ms for 10 FPS

    # 1. Coordinate alignment & direction vectors
    indices = np.searchsorted(data_t, acc_t)
    base_x = np.asarray(data_x)[indices]
    base_y = np.asarray(data_y)[indices]
    base_z = np.asarray(data_z)[indices]

    acc_x = np.asarray(acc_x)
    acc_y = np.asarray(acc_y)
    acc_z = np.asarray(acc_z)

    norms = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    norms_safe = np.where(norms == 0, 1e-8, norms)

    dir_x = acc_x / norms_safe
    dir_y = acc_y / norms_safe
    dir_z = acc_z / norms_safe

    # 2. Build Base Traces (Initial Frame t = 0)
    c_x, c_y, c_z, i_l, j_l, k_l = generate_cylinder_mesh(
        base_x[0],
        base_y[0],
        base_z[0],
        dir_x[0],
        dir_y[0],
        dir_z[0],
        rocket_radius,
        rocket_length,
    )

    fig = go.Figure()

    # Trace 0: Static Full Trajectory Path (Background)
    fig.add_trace(
        go.Scatter3d(
            x=data_x,
            y=data_y,
            z=data_z,
            mode="lines",
            line=dict(color="rgb(120, 120, 120)", width=2),
            customdata=data_t,
            hovertemplate=(
                "<b>Time:</b> %{customdata:.1f}s<br>"
                + "<b>X:</b> %{x:.2f}<br>"
                + "<b>Y:</b> %{y:.2f}<br>"
                + "<b>Z:</b> %{z:.2f}<extra></extra>"
            ),
            name="Trajectory",
        )
    )

    # Trace 1: Animated Rocket Body
    fig.add_trace(
        go.Mesh3d(
            x=c_x,
            y=c_y,
            z=c_z,
            i=i_l,
            j=j_l,
            k=k_l,
            color="silver",
            opacity=1.0,
            flatshading=True,
            name="Rocket Body",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Trace 2: Animated Thrust Vector Cone
    fig.add_trace(
        go.Cone(
            x=[base_x[0]],
            y=[base_y[0]],
            z=[base_z[0]],
            u=[acc_x[0]],
            v=[acc_y[0]],
            w=[acc_z[0]],
            sizemode="raw",
            sizeref=thrust_scale,
            anchor="tip",
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            showlegend=True,
            opacity=1.0,
            name="Thrust Vector",
        )
    )

    # Trace 3 (Optional): Static Keepout Polygon
    if polygon_coords is not None:
        poly_x = polygon_coords[:, 2]
        poly_y = polygon_coords[:, 1]
        poly_z = polygon_coords[:, 0]

        fig.add_trace(
            go.Mesh3d(
                x=poly_x,
                y=poly_y,
                z=poly_z,
                color="black",
                opacity=0.3,
                alphahull=0,
                name="Keepout Polygon",
                showlegend=True,
            )
        )

    # 3. Precompute Animation Frames
    frames = []
    slider_steps = []

    # Downsample frames to match the target real-time FPS
    # This prevents the browser from lagging on high-density data
    if len(acc_t) > 0:
        target_times = np.arange(acc_t[0], acc_t[-1], 1.0 / fps)
        frame_indices = np.searchsorted(acc_t, target_times)
        # Ensure we don't go out of bounds
        frame_indices = np.clip(frame_indices, 0, len(acc_t) - 1)
    else:
        frame_indices = []

    for i, k in enumerate(frame_indices):
        t_val = acc_t[k]
        frame_name = f"frame_{i}"

        cx, cy, cz, ci, cj, ck = generate_cylinder_mesh(
            base_x[k],
            base_y[k],
            base_z[k],
            dir_x[k],
            dir_y[k],
            dir_z[k],
            rocket_radius,
            rocket_length,
        )

        frame = go.Frame(
            name=frame_name,
            data=[
                # Update Rocket Mesh (Trace 1)
                go.Mesh3d(x=cx, y=cy, z=cz, i=ci, j=cj, k=ck),
                # Update Thrust Cone (Trace 2)
                go.Cone(
                    x=[base_x[k]],
                    y=[base_y[k]],
                    z=[base_z[k]],
                    u=[acc_x[k]],
                    v=[acc_y[k]],
                    w=[acc_z[k]],
                ),
            ],
            traces=[1, 2],  # Specify trace indices to update
        )
        frames.append(frame)

        # Slider step configuration
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [frame_name],
                    dict(
                        mode="immediate",
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=f"{t_val:.1f}s",
            )
        )

    fig.frames = frames

    # 4. Aspect Ratio and Static Bounding Box Calculations
    x_min, x_max = np.min(data_x), np.max(data_x)
    y_min, y_max = np.min(data_y), np.max(data_y)
    z_min, z_max = np.min(data_z), np.max(data_z)

    # Include rocket reach in padding
    pad = rocket_length + rocket_radius
    x_min, x_max = x_min - pad, x_max + pad
    y_min, y_max = y_min - pad, y_max + pad
    z_min, z_max = z_min - pad, z_max + pad

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    # 5. UI Controls (Play/Pause Buttons & Slider)
    fig.update_layout(
        title="Rocket Trajectory Animation",
        scene=dict(
            xaxis=dict(
                title="X Position", range=[x_mid - max_range, x_mid + max_range]
            ),
            yaxis=dict(
                title="Y Position", range=[y_mid - max_range, y_mid + max_range]
            ),
            zaxis=dict(
                title="Z Position", range=[z_mid - max_range, z_mid + max_range]
            ),
            bgcolor="rgb(250, 250, 250)",
            aspectmode="cube",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.00,
                y=0.05,
                xanchor="left",
                yanchor="top",
                pad=dict(t=20, r=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=frame_duration_ms, redraw=True
                                ),
                                fromcurrent=True,
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.18,
                y=0.05,
                len=0.78,
                xanchor="left",
                yanchor="top",
                pad=dict(t=20, b=10),
                currentvalue=dict(
                    font=dict(size=13),
                    prefix="Time: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0),
            )
        ],
    )

    fig.show()