import numpy as np
import scipy.stats as ss
from typing import Optional
from .ParticleTemplate import Particle


##################################
###### ConstAccelParticle2D ######

# 2D Particle with position, velocity and a constant acceleration
class ConstAccelParticle2D(Particle):

    def __init__(self):
        self.particle_dim = 6
    
    # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)

    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        return super().create_gaussian_particles(N, track_dim,  init_pos, std)
    
    # Constant acceleration prediction model using simple equations of motion.
    # We add random noise to acceleration to model non constant acceleration system.
    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, dt: float) -> np.ndarray:
        N = particles.shape[0]
        track_dim = particles.shape[1]

        # X positions
        particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt + np.random.randn(N, track_dim) * Q_model[:, 0]
        # X velocities
        particles[:, :, 1] += particles[:, :, 2] * dt + np.random.randn(N, track_dim) * Q_model[:, 1]
        # X accelerations
        particles[:, :, 2] += np.random.randn(N, track_dim) * Q_model[:, 2]
        
        # Y positions
        particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt + np.random.randn(N, track_dim) * Q_model[:, 3]
        # Y velocities
        particles[:, :, 4] += particles[:, :, 5] * dt + np.random.randn(N, track_dim) * Q_model[:, 4]
        # Y accelerations
        particles[:, :, 5] += np.random.randn(N, track_dim) * Q_model[:, 5]

        return particles
    
    # Measurement model using a similarity coefficient
    def measurement_model(self, coeff_sim: np.ndarray, R: np.ndarray) -> np.ndarray:
        proba = ss.norm(1., R[:, 0]).pdf(coeff_sim)
        return np.sum(proba, axis=1)