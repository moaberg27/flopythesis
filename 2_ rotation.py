# Here we build the rotation of the box 
import numpy as np

def calc_conductivity(k, theta, direction):
    
    # Convert theta to radians
    theta_radians = np.radians(theta)
    
    # Calculate the effective conductivity 
    k_eff = k * (np.cos(theta_radians)**2 + np.sin(theta_radians)**2) * (1-np.random.rand()/5)  # Adding some randomness to simulate variability
    
    return k_eff

rotations = [0, 15, 30, 45, 60, 75]

def compute_conductivity_points(k, rotations):
    points_x_pos = []
    points_y_pos = []
    points_x_neg = []
    points_y_neg = []

    for angle in rotations:
        kx = calc_conductivity(k, angle, direction='x') # conductivity in the x-direction
        kx_dir = np.exp(1j * np.radians(angle)) # direction of the conductivity in the complex plane
        
        kx_neg = calc_conductivity(k, angle + 180, direction='x') # conductivity in the opposite direction for x
        kx_neg_dir = np.exp(1j * np.radians(angle + 180)) # direction of the conductivity in the complex plane for the opposite direction for x
        
        ky = calc_conductivity(k, angle, direction='y') # conductivity in the y-direction
        ky_dir = np.exp(1j * np.radians(angle + 90)) # direction of the conductivity in the complex plane for the y-direction
        
        ky_neg = calc_conductivity(k, angle + 270, direction='y') # conductivity in the opposite direction for y
        ky_neg_dir = np.exp(1j * np.radians(angle + 270)) # direction of the conductivity in the complex plane for the opposite direction for y
        
        print (f"Angle: {angle} degrees")
        print (f"  kx: {kx:.3f}, direction: {np.degrees(np.angle(kx_dir)):.1f} degrees")
        print (f"  kx_neg: {kx_neg:.3f}, direction: {np.degrees(np.angle(kx_neg_dir)):.1f} degrees")
        print (f"  ky: {ky:.3f}, direction: {np.degrees(np.angle(ky_dir)):.1f} degrees")
        print (f"  ky_neg: {ky_neg:.3f}, direction: {np.degrees(np.angle(ky_neg_dir)):.1f} degrees")

        points_x_pos.append(kx * kx_dir) # effective conductivity in the x-direction
        points_y_pos.append(ky * ky_dir) # effective conductivity in the y-direction
        points_x_neg.append(kx_neg * kx_neg_dir) # effective conductivity in the opposite direction for x
        points_y_neg.append(ky_neg * ky_neg_dir) # effective conductivity in the opposite direction for y

    return points_x_pos, points_y_pos, points_x_neg, points_y_neg

points_x_pos, points_y_pos, points_x_neg, points_y_neg = compute_conductivity_points(1, rotations)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.plot(np.real(points_x_pos), np.imag(points_x_pos), 'o-', label='Effective Conductivity (x)')
plt.plot(np.real(points_y_pos), np.imag(points_y_pos), 's-', label='Effective Conductivity (y)')
plt.plot(np.real(points_x_neg), np.imag(points_x_neg), 'o-', label='Effective Conductivity (-x)')
plt.plot(np.real(points_y_neg), np.imag(points_y_neg), 's-', label='Effective Conductivity (-y)')
plt.xlabel('Conductivity in x-direction')
plt.ylabel('Conductivity in y-direction')
plt.title('Effective Conductivity with Rotation')
plt.legend()
plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.show()

# Fit 2D conductivity tensor from the 24 directional measurements 
# Each measurement satisfies: k_i = Kxx*cos²θ + 2*Kxy*cosθ*sinθ + Kyy*sin²θ

angles = []   # global measurement angles [rad]
k_meas = []   # measured conductivity magnitudes

for i, alpha in enumerate(rotations): # rotations (0, 15, 30, 45, 60, 75 degrees)
    # +x direction: global angle = alpha
    angles.append(np.radians(alpha)) 
    k_meas.append(abs(points_x_pos[i]))

    # -x direction: global angle = alpha + 180
    angles.append(np.radians(alpha + 180))
    k_meas.append(abs(points_x_neg[i]))

    # +y direction: global angle = alpha + 90
    angles.append(np.radians(alpha + 90))
    k_meas.append(abs(points_y_pos[i]))

    # -y direction: global angle = alpha + 270
    angles.append(np.radians(alpha + 270))
    k_meas.append(abs(points_y_neg[i]))

angles = np.array(angles) 
k_meas = np.array(k_meas)

# Build design matrix A so that A @ [Kxx, Kxy, Kyy] = k_meas
A = np.column_stack([
    np.cos(angles)**2,
    2 * np.cos(angles) * np.sin(angles),
    np.sin(angles)**2,
])

result, _, _, _ = np.linalg.lstsq(A, k_meas, rcond=None)
Kxx, Kxy, Kyy = result

K_tensor = np.array([[Kxx, Kxy],
                     [Kxy, Kyy]])

# Principal values and orientation via eigendecomposition
eigvals, eigvecs = np.linalg.eigh(K_tensor)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
principal_angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

# RMSE: how well the 24 points fit an ellipse (0 = perfect continuum)
k_pred = A @ result
rmse = np.sqrt(np.mean((k_meas - k_pred)**2))

print("\n--- Fitted 2D conductivity tensor ---")
print(f"  K = [[{Kxx:.4e}, {Kxy:.4e}],")
print(f"       [{Kxy:.4e}, {Kyy:.4e}]]")
print(f"  Principal values: k1 = {eigvals[0]:.4e}, k2 = {eigvals[1]:.4e}")
print(f"  Principal angle:  {principal_angle:.2f} deg")
print(f"  RMSE residual:    {rmse:.4e}  (0 = perfect ellipse)")

