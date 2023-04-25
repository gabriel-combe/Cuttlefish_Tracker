import numpy as np
import scipy.stats as ss
from typing import Optional, Tuple
from .ParticleTemplate import Particle


##################################
## ConstAccelParticle2DFixBbox ###

# 2D Particle with position, velocity, constant acceleration and fixed Bbox's width and height
class ConstAccelParticle2DFixBbox(Particle):

    def __init__(self, rng):
        self.particle_dim = 8
        self.rng = rng
    
    # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)

    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        return super().create_gaussian_particles(N, track_dim,  init_pos, std)
    
    # Constant acceleration prediction model using simple equations of motion.
    # We add random noise to acceleration to model non constant acceleration system.
    # We add random noise to width and height of the Bbox to model variation in Bbox size
    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, prev_particles: np.ndarray, frame_size: Tuple[int, int], dt: float) -> np.ndarray:
        N = particles.shape[0]
        track_dim = particles.shape[1]
        
        # Boxes width
        box_width = particles[:, :, 6]
        # Boxes height
        box_height = particles[:, :, 7]

        # X positions
        particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt
        # X velocities
        particles[:, :, 1] += particles[:, :, 2] * dt
        
        # Y positions
        particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt
        # Y velocities
        particles[:, :, 4] += particles[:, :, 5] * dt

        # Add Gaussian noise to the particles
        particles += self.rng.normal(loc=0, scale=Q_model, size=(N, track_dim, self.particle_dim))

        # Check constraints
        particles[:, :, 0] = np.maximum(0, np.minimum(frame_size[0], particles[:, :, 0])) # X positions
        particles[:, :, 3] = np.maximum(0, np.minimum(frame_size[1], particles[:, :, 3])) # Y positions
        particles[:, :, 6] = box_width # Boxes width
        particles[:, :, 7] = box_height # Boxes height

        return particles
    
    # Measurement model using a similarity coefficient
    def measurement_model(self, coeff_sim: np.ndarray, R: np.ndarray) -> np.ndarray:
        proba = ss.norm(0., R[:, 0]).pdf(coeff_sim)
        return proba


##################################
## ConstAccelParticle2DBbox ###

# 2D Particle with position, velocity, constant acceleration and Bbox's width and height
class ConstAccelParticle2DBbox(Particle):

    def __init__(self, rng):
        self.particle_dim = 8
        self.rng = rng
    
    # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)

    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        return super().create_gaussian_particles(N, track_dim,  init_pos, std)
    
    # Constant acceleration prediction model using simple equations of motion.
    # We add random noise to acceleration to model non constant acceleration system.
    # We add random noise to width and height of the Bbox to model variation in Bbox size
    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, prev_particles: np.ndarray, frame_size: Tuple[int, int], dt: float) -> np.ndarray:
        N = particles.shape[0]
        track_dim = particles.shape[1]

        # X positions
        particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt
        # X velocities
        particles[:, :, 1] += particles[:, :, 2] * dt
        
        # Y positions
        particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt
        # Y velocities
        particles[:, :, 4] += particles[:, :, 5] * dt

        # Add Gaussian noise to the particles
        particles += self.rng.normal(loc=0, scale=Q_model, size=(N, track_dim, self.particle_dim))

        # Check constraints
        particles[:, :, 0] = np.maximum(0, np.minimum(frame_size[0], particles[:, :, 0])) # X positions
        particles[:, :, 3] = np.maximum(0, np.minimum(frame_size[1], particles[:, :, 3])) # Y positions
        particles[:, :, 6] = np.maximum(particles[:, :, 6], 25) # Boxes width
        particles[:, :, 7] = np.maximum(particles[:, :, 7], 25) # Boxes height

        return particles
    
    # Measurement model using a similarity coefficient
    def measurement_model(self, coeff_sim: np.ndarray, R: np.ndarray) -> np.ndarray:
        proba = ss.norm(0., R[:, 0]).pdf(coeff_sim)
        return proba