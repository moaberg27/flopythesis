"""
Quick viewer for artifacts produced by dfn_2.py.
 
Paste one or more paths (from a run folder) when prompted and the matching
viewer opens:
  .vtk           -> PyVista interactive window
  .html          -> opened in the default browser
  .png/.jpg/...  -> matplotlib window
  .csv/.txt/.log -> printed to the terminal
  .npz           -> list arrays and print each one
 
Accepts multiple paths separated by ';' or newlines. Quotes around paths
(e.g. Windows Explorer's "Copy as path") are stripped automatically.
"""
 
import os
import webbrowser
 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
 
 
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
TEXT_EXTS  = {".csv", ".txt", ".log"}
 
 
def view_vtk(path):
    # pv.read returns the base DataObject type; cast via PolyData/wrap so the
    # type checker accepts it as a plottable mesh. dfn_2.py writes PolyData
    # shells, so this is a safe narrowing in practice.
    raw = pv.read(path)
    mesh = pv.wrap(raw) if not isinstance(raw, pv.DataSet) else raw
    plotter = pv.Plotter(title=os.path.basename(path))
    plotter.add_mesh(mesh, show_edges=True, opacity=0.6, color="lightblue")
    try:
        plotter.add_points(np.asarray(mesh.points),
                           color="darkred", point_size=6)
    except Exception:
        pass
    plotter.show_axes()  # type: ignore[call-arg]
    plotter.camera_position = "iso"
    plotter.show()
 
 
def view_html(path):
    webbrowser.open_new_tab("file:///" + os.path.abspath(path).replace("\\", "/"))
 
 
def view_image(path):
    img = mpimg.imread(path)
    _, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    ax.set_title(os.path.basename(path))
    ax.axis("off")
    plt.show(block=False)
 
 
def view_text(path, max_lines=200):
    print(f"\n===== {path} =====")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                print(f"... (truncated after {max_lines} lines)")
                break
            print(line.rstrip())
 
 
def view_npz(path):
    print(f"\n===== {path} =====")
    data = np.load(path, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        print(f"\n-- {key}  shape={arr.shape}  dtype={arr.dtype}")
        with np.printoptions(precision=4, suppress=True, threshold=50):
            print(arr)
 
 
def dispatch(path):
    path = path.strip().strip('"').strip("'")
    if not path:
        return
    if not os.path.exists(path):
        print(f"[missing] {path}")
        return
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".vtk":
            view_vtk(path)
        elif ext == ".html":
            view_html(path)
        elif ext in IMAGE_EXTS:
            view_image(path)
        elif ext in TEXT_EXTS:
            view_text(path)
        elif ext == ".npz":
            view_npz(path)
        else:
            print(f"[unsupported extension] {path}")
    except Exception as exc:
        print(f"[failed to open {path}] {type(exc).__name__}: {exc}")
 
 
def parse_paths(raw):
    """Split a pasted blob into individual paths.
 
    Users may paste multiple items at once separated by ';' or newlines
    (Windows Explorer multi-select with Shift+Right-click also joins paths
    with spaces inside quotes, which we handle by splitting on ';' first
    and trusting that any remaining spaces belong to a single path).
    """
    pieces = []
    for chunk in raw.replace("\r", "").split("\n"):
        for piece in chunk.split(";"):
            piece = piece.strip()
            if piece:
                pieces.append(piece)
    return pieces
 
 
if __name__ == "__main__":
    print("Paste one or more paths to view")
    while True:
        try:
            raw = input("\npath> ")
        except EOFError:
            break
        if not raw.strip():
            break
        for p in parse_paths(raw):
            dispatch(p)
        plt.show()  # block on any image windows before asking for the next path
 
 
 