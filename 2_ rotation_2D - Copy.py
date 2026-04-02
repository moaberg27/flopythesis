# Continuum assumption test for DFN models

# Input: 24 effective K values from a DFN model
#   - 6 rotations (0° to 75°, step 15°)
#   - 4 gradient BCs per rotation (W→E, E→W, S→N, N→S)

# Output: polar plot of K values + fitted conductivity ellipse
#   - If the 24 points fall on an ellipse → continuum assumption holds
#   - The fitted tensor can then be used for FLOPY upscaling

import numpy as np
import matplotlib.pyplot as plt

# Rotation angles (degrees)
rotations = [0, 15, 30, 45, 60, 75]

# DFN results collector -----------------------------------------------------------------------------------
# After each DFN rotation run, call add_rotation() with the 4 K values:
#   k_WE  = K from West→East gradient   (flow in +x direction)
#   k_EW  = K from East→West gradient   (flow in -x direction)
#   k_SN  = K from South→North gradient (flow in +y direction)
#   k_NS  = K from North→South gradient (flow in -y direction)

dfn_results = {} # defining a dictionary to store results for each rotation angle

# ADD ROTATION FUNCTION ---------------------------------------------------------------------------------------
# Stores the 4 K values from one DFN rotation into the dfn_results dictionary with the angle as the key, 
# and prints the added rotation and its K values for verification
def add_rotation(angle_deg, k_WE, k_EW, k_SN, k_NS):
    """Store the 4 K values from one DFN rotation."""
    dfn_results[angle_deg] = (k_WE, k_EW, k_SN, k_NS) # store results in dictionary with angle as key 
    print(f"  Added rotation {angle_deg} deg: " # print the added rotation and its K values for verification
          f"K(W>E)={k_WE:.4f}, K(E>W)={k_EW:.4f}, "
          f"K(S>N)={k_SN:.4f}, K(N>S)={k_NS:.4f}")
# ------------------------------------------------------------------------------------------------------------

# MOCK DFN FUNCTION ---------------------------------------------------------------------------------------
# It computes K for the 4 global directions that correspond to each BC experiment at a given rotation:
def mock_dfn(angle_deg, k_major=5.0, k_minor=1.0, principal_dir=30.0, noise=0.30):

    def _k_directional(global_angle_deg):
        theta = np.radians(global_angle_deg - principal_dir)
        k = k_major * np.cos(theta)**2 + k_minor * np.sin(theta)**2
        k *= (1.0 + noise * np.random.randn())  # add Gaussian noise
        return max(k, 1e-6)  # K must stay positive
    # ----------------------------------------------------------------------------
    # The 4 BC experiments probe these global directions:
    k_WE = _k_directional(angle_deg)         # W>E  = local +x = global angle
    k_EW = _k_directional(angle_deg + 180)   # E>W  = local -x = global angle+180
    k_SN = _k_directional(angle_deg + 90)    # S>N  = local +y = global angle+90
    k_NS = _k_directional(angle_deg + 270)   # N>S  = local -y = global angle+270
    return k_WE, k_EW, k_SN, k_NS
# ------------------------------------------------------------------------------------------------------------

# Run DFN simulations for all rotations and store results -------------------------------------------------------
for angle in rotations:
    # Call the mock DFN function to get K values for the 4 directions at this rotation angle, then add the results to the dfn_results dictionary using the add_rotation function
    k_WE, k_EW, k_SN, k_NS = mock_dfn(angle) # run the mock DFN function to get K values for the 4 directions at this rotation angle
    add_rotation(angle, k_WE, k_EW, k_SN, k_NS) # add the results to the dfn_results dictionary using the add_rotation function
# --------------------------------------------------------------------------------------------------------------

# Verify all rotations are present
assert set(dfn_results.keys()) == set(rotations), \
    f"Missing rotations: {set(rotations) - set(dfn_results.keys())}"

# Map DFN results to global directions -------------------------------------------------------
# At rotation angle α the model's local axes are rotated by α relative to
# the global coordinate system.  The four BC experiments probe:
#   W→E  → local +x → global direction = α
#   E→W  → local -x → global direction = α + 180°
#   S→N  → local +y → global direction = α + 90°
#   N→S  → local -y → global direction = α + 270°

points_x_pos = []  # W→E  (local +x)
points_x_neg = []  # E→W  (local -x)
points_y_pos = []  # S→N  (local +y)
points_y_neg = []  # N→S  (local -y)

# Loop through each rotation and extract the K values for the 4 directions,
# then convert them to complex points for plotting
for angle in rotations:
    k_WE, k_EW, k_SN, k_NS = dfn_results[angle]

    # W→E: magnitude k_WE plotted at global angle = α
    points_x_pos.append(k_WE * np.exp(1j * np.radians(angle)))

    # E→W: magnitude k_EW plotted at global angle = α + 180°
    points_x_neg.append(k_EW * np.exp(1j * np.radians(angle + 180)))

    # S→N: magnitude k_SN plotted at global angle = α + 90°
    points_y_pos.append(k_SN * np.exp(1j * np.radians(angle + 90)))

    # N→S: magnitude k_NS plotted at global angle = α + 270°
    points_y_neg.append(k_NS * np.exp(1j * np.radians(angle + 270)))

    print(f"Rotation {angle:2d} deg:  "
          f"K(W>E)={k_WE:.3f} @ {angle} deg,  "
          f"K(E>W)={k_EW:.3f} @ {angle+180} deg,  "
          f"K(S>N)={k_SN:.3f} @ {angle+90} deg,  "
          f"K(N>S)={k_NS:.3f} @ {angle+270} deg")
# -----------------------------------------------------------------------------------------------------------------

# Plot 1: raw 24 directional K measurements 
plt.figure(figsize=(7, 7))
plt.plot(np.real(points_x_pos), np.imag(points_x_pos), 'o', ms=8, label='W→E (+x)')
plt.plot(np.real(points_x_neg), np.imag(points_x_neg), 's', ms=8, label='E→W (−x)')
plt.plot(np.real(points_y_pos), np.imag(points_y_pos), '^', ms=8, label='S→N (+y)')
plt.plot(np.real(points_y_neg), np.imag(points_y_neg), 'v', ms=8, label='N→S (−y)')
plt.xlabel("K in x-direction")
plt.ylabel("K in y-direction")
plt.title("Directional Hydraulic Conductivity (24 DFN measurements)")
plt.legend()
plt.axis("equal")
all_x = np.concatenate([np.real(points_x_pos), np.real(points_x_neg), np.real(points_y_pos), np.real(points_y_neg)])
all_y = np.concatenate([np.imag(points_x_pos), np.imag(points_x_neg), np.imag(points_y_pos), np.imag(points_y_neg)])
shared_lim = max(np.abs(all_x).max(), np.abs(all_y).max()) * 1.15
plt.xlim(-shared_lim, shared_lim)
plt.ylim(-shared_lim, shared_lim)
plt.grid(True, alpha=0.3)
plt.tight_layout()


# Fit 2D conductivity tensor via least squares -------------------------------------------------
# Each measurement satisfies: k_i = Kxx·cos²θ + 2·Kxy·cosθ·sinθ + Kyy·sin²θ
# where θ is the global direction angle of that measurement.

angles = []   # global direction angles [rad]
k_meas = []   # measured K magnitudes

for i, alpha in enumerate(rotations): # extracts the raw data into two 1D lists: angles (global direction angles in radians) and k_meas (measured K magnitudes)
    # +x direction: global angle = alpha
    angles.append(np.radians(alpha)) # converts the rotation angle from degrees to radians and appends it to the angles list
    k_meas.append(abs(points_x_pos[i])) # appends the magnitude of the K value for the W→E direction (local +x) to the k_meas list

    # -x direction: global angle = alpha + 180
    angles.append(np.radians(alpha + 180))
    k_meas.append(abs(points_x_neg[i]))

    # +y direction: global angle = alpha + 90
    angles.append(np.radians(alpha + 90))
    k_meas.append(abs(points_y_pos[i]))

    # -y direction: global angle = alpha + 270
    angles.append(np.radians(alpha + 270))
    k_meas.append(abs(points_y_neg[i]))
# --------------------------------------------------------------------------------------------------

angles = np.array(angles)
k_meas = np.array(k_meas)

# Design matrix: A @ [Kxx, Kxy, Kyy] = k_meas -------------------------------
A = np.column_stack([
    np.cos(angles)**2,
    2 * np.cos(angles) * np.sin(angles),
    np.sin(angles)**2,
])

result, _, _, _ = np.linalg.lstsq(A, k_meas, rcond=None)
Kxx, Kxy, Kyy = result

K_tensor = np.array([[Kxx, Kxy],
                     [Kxy, Kyy]])

# Principal values and orientation via eigendecomposition ---------------------------------------
eigvals, eigvecs = np.linalg.eigh(K_tensor)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
principal_angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

# RMSE: how well the 24 points fit an ellipse (low = good continuum) -------------------------------
k_pred = A @ result
rmse = np.sqrt(np.mean((k_meas - k_pred)**2))
rel_rmse = rmse / np.mean(k_meas) * 100  # relative RMSE in %

print("\n" + "=" * 55)
print("  Fitted 2D Conductivity Tensor (for FLOPY upscaling)")
print(f"  K = [[{Kxx:.4e}, {Kxy:.4e}],")
print(f"       [{Kxy:.4e}, {Kyy:.4e}]]")
print(f"  Principal values: k1 = {eigvals[0]:.4e},  k2 = {eigvals[1]:.4e}")
print(f"  Principal angle:  {principal_angle:.2f} deg")
print(f"  Anisotropy ratio: {eigvals[0]/eigvals[1]:.2f}")
print(f"  RMSE residual:    {rmse:.4e}  (relative: {rel_rmse:.2f}%)")
if rel_rmse < 5:
    print("  --> Good ellipse fit: continuum assumption likely valid")
elif rel_rmse < 15:
    print("  --> Moderate fit: continuum assumption approximate")
else:
    print("  --> Poor fit: continuum assumption may NOT hold")
print("=" * 55)


# Plot 2: measurements + fitted ellipse + principal axes -------------------------------------------------
theta_fit = np.linspace(0, 2 * np.pi, 200)
ellipse_x = (eigvals[0] * np.cos(theta_fit) * np.cos(np.radians(principal_angle))
           - eigvals[1] * np.sin(theta_fit) * np.sin(np.radians(principal_angle)))
ellipse_y = (eigvals[0] * np.cos(theta_fit) * np.sin(np.radians(principal_angle))
           + eigvals[1] * np.sin(theta_fit) * np.cos(np.radians(principal_angle)))

plt.figure(figsize=(7, 7))
plt.plot(np.real(points_x_pos), np.imag(points_x_pos), 'o', ms=8, label='W→E (+x)')
plt.plot(np.real(points_x_neg), np.imag(points_x_neg), 's', ms=8, label='E→W (−x)')
plt.plot(np.real(points_y_pos), np.imag(points_y_pos), '^', ms=8, label='S→N (+y)')
plt.plot(np.real(points_y_neg), np.imag(points_y_neg), 'v', ms=8, label='N→S (−y)')
plt.plot(ellipse_x, ellipse_y, 'k--', lw=1.5, label='Fitted ellipse')

# Principal axes (length proportional to eigenvalue)
origin = [0], [0]
plt.quiver(*origin, eigvals[0]*eigvecs[0, 0], eigvals[0]*eigvecs[1, 0],
           color='r', scale=1, scale_units='xy', angles='xy',
           label=f'k1 = {eigvals[0]:.2f}')
plt.quiver(*origin, eigvals[1]*eigvecs[0, 1], eigvals[1]*eigvecs[1, 1],
           color='b', scale=1, scale_units='xy', angles='xy',
           label=f'k2 = {eigvals[1]:.2f}')

plt.text(0.05, 0.95,
         f"Principal values:\n$k_1$ = {eigvals[0]:.3f}\n$k_2$ = {eigvals[1]:.3f}\n"
         f"Angle = {principal_angle:.1f}°\nRMSE = {rel_rmse:.1f}%",
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.xlabel("K in x-direction")
plt.ylabel("K in y-direction")
plt.title("Directional K with Fitted Conductivity Ellipse")
plt.legend(loc='lower right')
plt.axis("equal")
plt.xlim(-shared_lim, shared_lim)
plt.ylim(-shared_lim, shared_lim)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
