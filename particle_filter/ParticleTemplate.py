import numpy as np
from typing import Optional, Tuple

# Template for Particle classes
class Particle(object):
    def __init__(self, rng):
        self.particle_dim: int = 0  # Dimension of the state vector
        self.rng = rng
    
    # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        return self.rng.uniform(low=ranges[:, :, 0], high=ranges[:, :, 1], size=(N, track_dim, self.particle_dim))

    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        return self.rng.normal(loc=init_pos, scale=std, size=(N, track_dim, self.particle_dim))
    
    # Motion model used to predict one step of the particle
    # Need to be implemented for each Particle object
    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, prev_particles: np.ndarray, frame_size: Tuple[int, int], dt: float) -> np.ndarray:
        pass

    # Measurement model used to get the probability 
    # that the particle match the real system
    # Need to be implemented for each Particle object
    def measurement_model(self, coeff_sim: np.ndarray, R: np.ndarray) -> np.ndarray:
        pass