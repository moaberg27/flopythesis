"""
rotation_state_viewer.py
------------------------
Minimal viewer using matplotlib. Type a state number (1-266) in the
text box and press Enter to see the rotated cube.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import TextBox


# ── Helpers ──────────────────────────────────────────────────────────────────

def rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    else:  # 'z'
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])


FACE_NORMALS_LOCAL = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)


def precompute_directions(rotations):
    step = rotations[1] - rotations[0]
    seen, dirs, states = set(), [], []
    for phi in range(0, 360, step):
        for theta in range(0, 180, step):
            R = rotation_matrix('x', theta) @ rotation_matrix('z', phi)
            for fi, n_local in enumerate(FACE_NORMALS_LOCAL):
                n_lab = R @ n_local
                key = tuple(np.round(n_lab, 8))
                if key not in seen:
                    seen.add(key)
                    dirs.append(n_lab)
                    states.append((phi, theta, fi))
    return np.array(dirs), states


rotations = [0, 15, 30, 45, 60, 75]
all_directions, all_states = precompute_directions(rotations)
N = len(all_states)
print(f"Total unique rotation states: {N}")


# ── Cube geometry ────────────────────────────────────────────────────────────

_corners = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
], dtype=float) * 0.5

# 6 faces as vertex index lists
_faces = [
    [1,2,6,5],  # +X
    [0,3,7,4],  # -X
    [2,3,7,6],  # +Y
    [0,1,5,4],  # -Y
    [4,5,6,7],  # +Z
    [0,1,2,3],  # -Z
]

_face_labels = ['+X','-X','+Y','-Y','+Z','-Z']
_face_normals_6 = np.array([
    [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)


# ── Plot setup ───────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(9, 8))
ax = fig.add_axes([0.05, 0.12, 0.9, 0.82], projection='3d')

# text box at the bottom
ax_box = fig.add_axes([0.3, 0.02, 0.15, 0.04])
text_box = TextBox(ax_box, f'State (1-{N}):  ', initial='1')


def draw_state(state_idx):
    ax.cla()

    phi, theta, fi = all_states[state_idx]
    R = rotation_matrix('x', theta) @ rotation_matrix('z', phi)
    n_lab = all_directions[state_idx]
    face_name = 'XYZ'[fi]

    # ── Fixed global axes ──
    L = 1.0
    for i, (col, lbl) in enumerate(zip(['r','g','b'], ['X','Y','Z'])):
        v = np.zeros(3); v[i] = L
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=2)
        ax.text(v[0]*1.1, v[1]*1.1, v[2]*1.1, lbl, color=col,
                fontsize=12, fontweight='bold')

    # ── Rotated cube ──
    pts = (R @ _corners.T).T
    polys = [[pts[j] for j in face] for face in _faces]
    cube = Poly3DCollection(polys, alpha=0.15, facecolor='lightgrey',
                            edgecolor='black', linewidth=1)
    ax.add_collection3d(cube)

    # face labels
    for normal, lbl in zip(_face_normals_6, _face_labels):
        pos = R @ (normal * 0.55)
        ax.text(pos[0], pos[1], pos[2], lbl, color='dimgrey',
                fontsize=8, ha='center', va='center')

    # ── Measurement direction ──
    ax.quiver(0, 0, 0, n_lab[0], n_lab[1], n_lab[2],
              color='goldenrod', arrow_length_ratio=0.12, linewidth=2.5)
    tip = n_lab * 1.2
    ax.text(tip[0], tip[1], tip[2], f'n ({face_name})',
            color='goldenrod', fontsize=10, fontweight='bold')

    # ── Formatting ──
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f"State {state_idx+1}/{N}    "
        f"phi={phi}    theta={theta}    face={fi} ({face_name})",
        fontsize=11)

    fig.canvas.draw_idle()


def on_submit(text):
    text = text.strip()
    try:
        idx = int(text) - 1
    except ValueError:
        return
    if 0 <= idx < N:
        draw_state(idx)


text_box.on_submit(on_submit)
draw_state(0)
plt.show()