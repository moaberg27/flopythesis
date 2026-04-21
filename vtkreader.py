import pyvista as pv

def view_vtk_file(file_path):
    """
    Visualize a VTK file using PyVista.

    Parameters:
        file_path (str): The path to the VTK file to visualize.
    """
    try:
        # Load the VTK file
        mesh = pv.read(file_path)

        # Check if the mesh is a valid DataSet or MultiBlock
        if not isinstance(mesh, (pv.DataSet, pv.MultiBlock)):
            raise TypeError("The file does not contain a valid DataSet or MultiBlock for visualization.")

        # Create a plotter
        plotter = pv.Plotter()

        # Add the mesh to the plotter
        plotter.add_mesh(mesh, show_edges=True)

        # Display the plot
        plotter.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# Replace the path below with your VTK file path
view_vtk_file(r"C:\Users\SEAM94860\FLOPY\finalflopy\runs\250box_15degree_2nd\20260420_155522\shell.vtk")