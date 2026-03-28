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
