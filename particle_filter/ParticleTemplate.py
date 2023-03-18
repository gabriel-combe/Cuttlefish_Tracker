import numpy as np
from typing import Optional

# Template for Particle classes
class Particle(object):
    def __init__(self):
        
        self.particle_dim: int = 0 # Dimension of the state vector
    
    # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = np.random.uniform(ranges[:, i, 0], ranges[:, i, 1], size=(N, track_dim))
        return particles

    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = init_pos[:, i] + (np.random.randn(N, track_dim) * std[:, i])
        return particles
    
    # Motion model used to predict one step of the particle
    # Need to be implemented for each Particle object
    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, dt: float) -> np.ndarray:
        pass

    # Measurement model used to get the probability 
    # that the particle match the real system
    # Need to be implemented for each Particle object
    def measurement_model(self, coeff_sim: np.ndarray) -> np.ndarray:
        pass