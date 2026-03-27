# Here we build the rotation of the box 
import numpy as np

def calc_conductivity(k, theta, direction='x'):
    
    # Convert theta to radians
    theta_rad = np.radians(theta)
    
    # Calculate the effective conductivity using the rule of mixtures
    k_eff = k * (np.cos(theta_rad)**2 + np.sin(theta_rad)**2) * (1-np.random.rand()/5)  # Adding some randomness to simulate variability
    
    return k_eff

rotations = [0, 15, 30, 45, 60, 75]

pntsx = []
pnty = []

#TODO: wirte into a function
#TODO: only rotate 4 times
for r in rotations:
    kx = calc_conductivity(1, r, direction='x')
    kx_vec = np.exp(1j * np.radians(r))
    kxm = calc_conductivity(1, r+180, direction='x')
    ky = calc_conductivity(1, r, direction='y')
    ky_vec = np.exp(1j * np.radians(r + 90))
    pntsx.append(kx * kx_vec)
    pnty.append(ky * ky_vec)
    
# TODO: plot
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.plot(np.real(pntsx), np.imag(pntsx), 'o-', label='Effective Conductivity')
plt.plot(np.real(pnty), np.imag(pnty), 's-', label='Effective Conductivity (y)')
plt.xlabel('Conductivity in x-direction')
plt.ylabel('Conductivity in y-direction')
plt.title('Effective Conductivity with Rotation')
plt.legend()
plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.show()
