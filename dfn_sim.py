
"""
Notes
-----
This is an example of a model.
"""

import datetime
import os

import numpy as np
import pandas as pd

import andfn

# Configure logging
import logging

from andfn import ConstantHeadLine

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # save figure
    save = False
    scale = 1
    tracking = True
    animate = False

    ncoef = 10 * 0 + 10
    nint = ncoef * 2

    start0 = datetime.datetime.now()
    print("\n---- IMPORT DFN ----")
    print(f"Program started at {start0}")

    # load the geometry
    dfn_org = andfn.DFN("DFN test FracMan", discharge_int=50)

    # name ="p32_case11"
    path = os.path.join("fracs", "fracs_connected_properties.csv")
    #path = os.path.join(r"C:\Users\seet92866\PycharmProjects\DFN_exjobb\2", "15000fracs.csv")
    reload = False

    # Check if it exist saved
    print("DFN  importing from file")
    dfn_org.import_fractures_from_file(
        path,
        radius_str="EquivRadius[m]",
        x_str="FractureX[m]",
        y_str="FractureY[m]",
        z_str="FractureZ[m]",
        t_str="Transmissivity[m2/s]",
        trend_str="Trend[deg]",
        plunge_str="Plunge[deg]",
        e_str="Aperture[m]",
        remove_tolerance=1e-3,
        remove_isolated=False
    )


    # TODO: add the rotation loop
    for r in range(1):
        dfn = andfn.DFN("Copy", discharge_int=50)
        dfn.add_fracture(dfn_org.fractures)

        print("Adding constant head boundary conditions")
        # Add Region Box boundary
        head0 = 100
        head1 = 200
        regbox = andfn.RectangularRegion(
            label="box",
            center=[0, 0, 0],
            x_vec=[1, 0, 0],
            y_vec=[0, 1, 0],
            z_vec=[0, 0, 1],
            xl=500,
            yl=500,
            zl=500,
        )
        regbox.rotate(angle=45, axis=[1, 0, 0])
        regbox.rotate(angle=45, axis=[0, 0, 1])
        reg_fracs_in, reg_fracs_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)

        print(f"Number of fractures in the DFN: {len(dfn.fractures)}")
        dfn.delete_fracture(reg_fracs_out)
        print(f"Number of fractures after deleting those outside the box: {len(dfn.fractures)}")

        regbox.frac_intersections(dfn.fractures, face="left", head=head0)
        regbox.frac_intersections(dfn.fractures, face="right", head=head1)

        dfn.check_connectivity()

        dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30, MAX_NCOEF=200, MAX_ERROR=5e-4)

        start1 = datetime.datetime.now()
        print("\n---- SOLVE THE DFN ----")
        dfn.solve(unconsolidate=True)

        start2 = datetime.datetime.now()

        print("\n---- GET FLOWS ----")
        total_flow = [
            np.abs(e.q) for e in regbox.elements if isinstance(e, ConstantHeadLine)
        ]
        sum_flows = np.sum(total_flow) / 2
        print(f"Total flow through the box: {sum_flows:.2e} m^3/s")

        print("\n---- PLOTTING ----")
        p1 = dfn.initiate_plotter(title=True, off_screen=False, scale=1, axis=True)

        dfn.plot_fractures_head(
            p1, 40, 10, opacity=1, contour=True
        )  # , limits=[200, 400], debug=False)
        regbox.plot(p1)



        p1.show()

    end = datetime.datetime.now()
    print(f"\n\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
    print(f"\t-generating: \t{start1 - start0}")
    print(f"\t-solving: \t\t\t{start2 - start1}")
    print(f"\t-plotting: \t\t{end - start2}")

    print("All done!")
