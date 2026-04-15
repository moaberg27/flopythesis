"""Interactive rotation visualisation based on data generated in ny_rotation.py.

Features:
- Box drawn in global coordinate system
- Animation over all rotation labels found in ny_rotation.py
- Pause/Play button
- Text box to jump to a specific rotation index
- Highlights points added in the current rotation step
"""

from pathlib import Path
import importlib.util
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, TextBox


ROOT = Path(__file__).resolve().parent
NY_ROTATION_PATH = ROOT / "ny_rotation.py"


def load_data_from_ny_rotation(script_path: Path):
    """Execute ny_rotation.py in an isolated module and return computed arrays."""
    if not script_path.exists():
        raise FileNotFoundError(f"Missing file: {script_path}")

    # Avoid opening ny_rotation's own plot when importing its module-level script.
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        spec = importlib.util.spec_from_file_location(
            "ny_rotation_data", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        plt.show = original_show
        plt.close("all")

    required_attrs = ["points", "rot_ids", "face_ids",
                      "rotation_matrix_zxy", "FACE_COLORS"]
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise AttributeError(
                f"ny_rotation.py saknar variabel/funktion: {attr}")

    points = np.asarray(module.points, dtype=float)
    rot_ids = np.asarray(module.rot_ids).astype(str)
    face_ids = np.asarray(module.face_ids).astype(str)

    return {
        "points": points,
        "rot_ids": rot_ids,
        "face_ids": face_ids,
        "rotation_matrix_zxy": module.rotation_matrix_zxy,
        "face_colors": dict(module.FACE_COLORS),
    }


def build_box_geometry(side_length=1.0):
    """Create box vertices and edges centered at origin."""
    h = side_length / 2.0
    vertices = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return vertices, edges


def build_box_faces():
    """Return face name to vertex-index mapping for the box."""
    return {
        "+x": [1, 2, 6, 5],
        "-x": [0, 3, 7, 4],
        "+y": [2, 3, 7, 6],
        "-y": [0, 1, 5, 4],
        "+z": [4, 5, 6, 7],
        "-z": [0, 1, 2, 3],
    }


def parse_rotation_matrix(rot_label: str, rotation_matrix_zxy):
    """Map ny_rotation label to its corresponding orientation matrix."""
    if rot_label == "start":
        return np.eye(3)

    m = re.match(r"base_z(?P<z>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        return rotation_matrix_zxy(float(m.group("z")), 0.0, 0.0)

    m = re.match(r"tilt_y(?P<y>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        return rotation_matrix_zxy(0.0, 0.0, float(m.group("y")))

    m = re.match(
        r"tilt_y(?P<y>-?\d+(?:\.\d+)?)_z(?P<z>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        r_tilt = rotation_matrix_zxy(0.0, 0.0, float(m.group("y")))
        r_xy = rotation_matrix_zxy(float(m.group("z")), 0.0, 0.0)
        return r_xy @ r_tilt

    m = re.match(r"step3_yz_x(?P<x>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        return rotation_matrix_zxy(0.0, float(m.group("x")), 0.0)

    m = re.match(
        r"step3_yz_x(?P<x>-?\d+(?:\.\d+)?)_xy_z(?P<z>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        r_yz = rotation_matrix_zxy(0.0, float(m.group("x")), 0.0)
        r_xy = rotation_matrix_zxy(float(m.group("z")), 0.0, 0.0)
        return r_xy @ r_yz

    m = re.match(r"step4_yz_x(?P<x>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        return rotation_matrix_zxy(0.0, float(m.group("x")), 0.0)

    m = re.match(
        r"step4_yz_x(?P<x>-?\d+(?:\.\d+)?)_xy_z(?P<z>-?\d+(?:\.\d+)?)$", rot_label)
    if m:
        r_yz = rotation_matrix_zxy(0.0, float(m.group("x")), 0.0)
        r_xy = rotation_matrix_zxy(float(m.group("z")), 0.0, 0.0)
        return r_xy @ r_yz

    return np.eye(3)


def main():
    data = load_data_from_ny_rotation(NY_ROTATION_PATH)
    points = data["points"]
    rot_ids = data["rot_ids"]
    face_ids = data["face_ids"]
    rotation_matrix_zxy = data["rotation_matrix_zxy"]
    face_colors = data["face_colors"]

    rotation_labels = list(dict.fromkeys(rot_ids.tolist()))
    if not rotation_labels:
        raise ValueError("Inga rotationer hittades i ny_rotation.py")

    label_to_indices = {label: np.where(rot_ids == label)[
        0] for label in rotation_labels}

    vertices, edges = build_box_geometry(side_length=1.0)
    face_to_vertices = build_box_faces()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2)

    lim = max(1.2, np.max(np.abs(points)) * 1.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("Kx")
    ax.set_ylabel("Ky")
    ax.set_zlabel("Kz")

    # Global coordinate frame.
    ax.plot([-lim, lim], [0, 0], [0, 0],
            color="gray", linewidth=1.0, alpha=0.3)
    ax.plot([0, 0], [-lim, lim], [0, 0],
            color="gray", linewidth=1.0, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-lim, lim],
            color="gray", linewidth=1.0, alpha=0.3)

    edge_lines = [ax.plot([], [], [], color="#1f77b4",
                          linewidth=2.0)[0] for _ in edges]

    # Transparent face colors based on ny_rotation FACE_COLORS.
    face_patches = {}
    for face_name in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        idx = face_to_vertices[face_name]
        face_xyz = [vertices[i] for i in idx]
        patch = Poly3DCollection(
            [face_xyz],
            facecolors=face_colors.get(face_name, "#aaaaaa"),
            edgecolors="none",
            alpha=0.20,
        )
        ax.add_collection3d(patch)
        face_patches[face_name] = patch

    state = {
        "frame_idx": 0,
        "paused": False,
        "scatters": [],
        "texts": [],
    }

    info_text = ax.text2D(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    def clear_dynamic_artists():
        for artist in state["scatters"]:
            artist.remove()
        state["scatters"].clear()

        for artist in state["texts"]:
            artist.remove()
        state["texts"].clear()

    def draw_frame(frame_idx: int):
        frame_idx = int(frame_idx) % len(rotation_labels)
        state["frame_idx"] = frame_idx

        label = rotation_labels[frame_idx]
        r = parse_rotation_matrix(label, rotation_matrix_zxy)
        rotated = vertices @ r.T

        for line, (i, j) in zip(edge_lines, edges):
            xyz = rotated[[i, j]]
            line.set_data(xyz[:, 0], xyz[:, 1])
            line.set_3d_properties(xyz[:, 2])

        for face_name, idx in face_to_vertices.items():
            face_xyz = [rotated[i] for i in idx]
            face_patches[face_name].set_verts([face_xyz])

        clear_dynamic_artists()

        # Cumulative points in faint gray.
        cum_idx = np.concatenate([label_to_indices[lbl]
                                 for lbl in rotation_labels[: frame_idx + 1]])
        cum_pts = points[cum_idx]
        cum_faces = face_ids[cum_idx]
        cum_colors = [face_colors.get(face, "#bbbbbb") for face in cum_faces]
        scatter_cum = ax.scatter(
            cum_pts[:, 0],
            cum_pts[:, 1],
            cum_pts[:, 2],
            c=cum_colors,
            s=18,
            alpha=0.15,
            depthshade=False,
        )
        state["scatters"].append(scatter_cum)

        # New points for current rotation in face color.
        new_idx = label_to_indices[label]
        new_pts = points[new_idx]
        new_faces = face_ids[new_idx]
        new_colors = [face_colors.get(face, "#222222") for face in new_faces]

        scatter_new = ax.scatter(
            new_pts[:, 0],
            new_pts[:, 1],
            new_pts[:, 2],
            c=new_colors,
            s=70,
            alpha=0.35,
            edgecolors="black",
            linewidths=0.4,
            depthshade=False,
        )
        state["scatters"].append(scatter_new)

        for p, face in zip(new_pts, new_faces):
            t = ax.text(
                p[0] + 0.03,
                p[1] + 0.03,
                p[2] + 0.03,
                face,
                fontsize=8,
                alpha=0.9,
            )
            state["texts"].append(t)

        info_text.set_text(
            "Rotation {}/{}\n".format(frame_idx + 1, len(rotation_labels))
            + f"label: {label}\n"
            + f"new points this step: {len(new_idx)}"
        )

        return [*edge_lines, *face_patches.values(), info_text, *state["scatters"], *state["texts"]]

    def animate(_):
        if not state["paused"]:
            state["frame_idx"] = (state["frame_idx"] +
                                  1) % len(rotation_labels)
        return draw_frame(state["frame_idx"])

    # Controls
    pause_ax = fig.add_axes([0.10, 0.06, 0.12, 0.06])
    pause_btn = Button(pause_ax, "Pause")

    jump_ax = fig.add_axes([0.29, 0.06, 0.42, 0.06])
    jump_box = TextBox(jump_ax, "Go to rotation # ", initial="1")

    fig.text(0.75, 0.065, "Space = pause/play", fontsize=9)

    def toggle_pause(_event=None):
        state["paused"] = not state["paused"]
        pause_btn.label.set_text("Play" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    def on_submit(text):
        try:
            requested = int(text.strip())
        except ValueError:
            return

        if 1 <= requested <= len(rotation_labels):
            state["frame_idx"] = requested - 1
            draw_frame(state["frame_idx"])
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            toggle_pause()

    pause_btn.on_clicked(toggle_pause)
    jump_box.on_submit(on_submit)
    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_title("Rotation visualisation from ny_rotation.py")
    draw_frame(0)
    anim = FuncAnimation(fig, animate, interval=280, blit=False)

    _ = anim
    plt.show()


if __name__ == "__main__":
    main()
