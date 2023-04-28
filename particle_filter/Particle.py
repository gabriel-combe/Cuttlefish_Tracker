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
    def motion_model(self, 
        particles: np.ndarray, Q_model: np.ndarray, 
        prev_particles: np.ndarray, template_particle: np.ndarray,
        search_area: Tuple[int, int], frame_size: Tuple[int, int], dt: float) -> np.ndarray:

        N = particles.shape[0]
        track_dim = particles.shape[1]
        
        # Boxes width
        box_width = particles[:, :, 6]
        # Boxes height
        box_height = particles[:, :, 7]

        # X positions
        particles[:, :, 0] += .5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt
        # X velocities
        particles[:, :, 1] += particles[:, :, 2] * dt
        
        # Y positions
        particles[:, :, 3] += .5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt
        # Y velocities
        particles[:, :, 4] += particles[:, :, 5] * dt

        # Add Gaussian noise to the particles
        noise = self.rng.normal(loc=0, scale=Q_model, size=(N, track_dim, self.particle_dim))
        particles[:, :, [0, 1, 2, 3, 4, 5]] += noise[:, :, [0, 1, 2, 3, 4, 5]]

        # Check constraints
        # Check if position x is within 3/4 of the search area
        indices_x = (particles[:, :, 0] > template_particle[:, 0] + 3*search_area[:, 0]/4) | (particles[:, :, 0] < template_particle[:, 0] - 3*search_area[:, 0]/4) |\
                    (particles[:, :, 0] >= frame_size[0]) | (particles[:, :, 0] < 0)
        particles[indices_x] = template_particle
        
        # Check if position y is within 3/4 of the search area
        indices_y = (particles[:, :, 3] > template_particle[:, 3] + 3*search_area[:, 1]/4) | (particles[:, :, 3] < template_particle[:, 3] - 3*search_area[:, 1]/4) |\
                    (particles[:, :, 3] >= frame_size[1]) | (particles[:, :, 3] < 0)
        particles[indices_y] = template_particle
        particles[:, :, 6] = box_width # Boxes width
        particles[:, :, 7] = box_height # Boxes height

        return particles
    
    # Measurement model using a similarity coefficient
    def measurement_model(self, coeff_sim: np.ndarray, particles: np.ndarray, template_particle:np.ndarray, R: np.ndarray) -> np.ndarray:
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
    
    def compute_gamma(self, N: int, track_dim: int) -> np.ndarray:
        p = self.rng.random((N, track_dim))
        gamma = np.zeros((N, track_dim))
        gamma[(0 <= p) & (p <= 0.2)] = -0.05
        gamma[(0.2 < p) & (p <= 0.4)] = -0.025
        gamma[(0.6 < p) & (p <= 0.8)] = 0.025
        gamma[(0.8 < p) & (p <= 1)] = 0.05

        return gamma
    
    # Constant acceleration prediction model using simple equations of motion.
    # We add random noise to acceleration to model non constant acceleration system.
    # We add random noise to width and height of the Bbox to model variation in Bbox size
    def motion_model(self, 
        particles: np.ndarray, Q_model: np.ndarray, 
        prev_particles: np.ndarray, template_particle: np.ndarray,
        search_area: np.ndarray, frame_size: Tuple[int, int], dt: float) -> np.ndarray:

        N = particles.shape[0]
        track_dim = particles.shape[1]
        gamma = self.compute_gamma(N, track_dim)

        # X positions
        particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt
        # X velocities
        particles[:, :, 1] += particles[:, :, 2] * dt
        
        # Y positions
        particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt
        # Y velocities
        particles[:, :, 4] += particles[:, :, 5] * dt

        # Boxes width
        particles[:, :, 6] *= 1 + gamma
        # Boxes height
        particles[:, :, 7] *= 1 + gamma

        # Add Gaussian noise to the particles
        noises = self.rng.normal(loc=0, scale=Q_model, size=(N, track_dim, self.particle_dim))
        particles += noises

        # Check constraints
        # Check if position x is within 3/4 of the search area
        indices_x = (particles[:, :, 0] > template_particle[:, 0] + 3*search_area[:, 0]/4) | (particles[:, :, 0] < template_particle[:, 0] - 3*search_area[:, 0]/4) |\
                    (particles[:, :, 0] >= frame_size[0]) | (particles[:, :, 0] < 0)
        particles[indices_x] = template_particle
        
        # Check if position y is within 3/4 of the search area
        indices_y = (particles[:, :, 3] > template_particle[:, 3] + 3*search_area[:, 1]/4) | (particles[:, :, 3] < template_particle[:, 3] - 3*search_area[:, 1]/4) |\
                    (particles[:, :, 3] >= frame_size[1]) | (particles[:, :, 3] < 0)
        particles[indices_y] = template_particle
        
        # Check if bbox is between the size of the search area and 16
        particles[:, :, 6] = np.minimum(search_area[:, 0], np.maximum(particles[:, :, 6], 16)) # Boxes width
        particles[:, :, 7] = np.minimum(search_area[:, 1], np.maximum(particles[:, :, 7], 16)) # Boxes height

        return particles
    
    # Measurement model using a similarity coefficient
    def measurement_model(self, coeff_sim: np.ndarray, particles: np.ndarray, template_particle:np.ndarray, R: np.ndarray) -> np.ndarray:
        proba = ss.norm.pdf(coeff_sim, loc=0, scale=R[:, 0])
        # proba = 0.8*ss.norm.pdf(coeff_sim, loc=0, scale=R[:, 0]) + (1-0.8)*ss.norm.pdf(np.mean((particles - template_particle)**2, axis=(2, 1)), loc=0, scale=R[:, 1])
        # proba = np.exp(-R[:, 0] * coeff_sim**2)
        return proba


##################################
##### PredPosParticle2DBbox ######

# 2D Particle with position, Bbox's width and height 
class PredPosParticle2DBbox(Particle):

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
    
    def compute_gamma(self, N: int, track_dim: int) -> np.ndarray:
        p = self.rng.random((N, track_dim))
        gamma = np.zeros((N, track_dim))
        gamma[(0 <= p) & (p <= 0.2)] = -0.05
        gamma[(0.2 < p) & (p <= 0.4)] = -0.025
        gamma[(0.6 < p) & (p <= 0.8)] = 0.025
        gamma[(0.8 < p) & (p <= 1)] = 0.05

        return gamma
    
    # Constant acceleration prediction model using finite differences equations.
    # We add random noise to model non linear system.
    # We add random noise to width and height of the Bbox to model variation in Bbox size
    def motion_model(self, 
        particles: np.ndarray, Q_model: np.ndarray, 
        prev_particles: np.ndarray, template_particle: np.ndarray,
        search_area: Tuple[int, int], frame_size: Tuple[int, int], dt: float) -> np.ndarray:

        N = particles.shape[0]
        track_dim = particles.shape[1]
        gamma = self.compute_gamma(N, track_dim)

        # X velocities
        particles[:, :, 1] = (particles[:, :, 0] - prev_particles[0, :, :, 0]) / dt
        # X acceleration
        particles[:, :, 2] = (particles[:, :, 1] - prev_particles[1, :, :, 1]) / dt
        # X positions
        particles[:, :, 0] += .5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt
        
        # X velocities
        particles[:, :, 4] = (particles[:, :, 3] - prev_particles[0, :, :, 3]) / dt
        # X acceleration
        particles[:, :, 5] = (particles[:, :, 4] - prev_particles[1, :, :, 4]) / dt
        # Y positions
        particles[:, :, 3] += .5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt

        # Boxes width
        particles[:, :, 6] *= 1 + gamma
        # Boxes height
        particles[:, :, 7] *= 1 + gamma

        # Add Gaussian noise to the particles
        noises = self.rng.normal(loc=0, scale=Q_model, size=(N, track_dim, self.particle_dim))
        particles[:, :, [0, 3, 6, 7]] += noises[:, :, [0, 3, 6, 7]]

        # Check constraints
        # Check if position x is within 3/4 of the search area
        indices_x = (particles[:, :, 0] > template_particle[:, 0] + 3*search_area[:, 0]/4) | (particles[:, :, 0] < template_particle[:, 0] - 3*search_area[:, 0]/4) |\
                    (particles[:, :, 0] >= frame_size[0]) | (particles[:, :, 0] < 0)
        particles[indices_x] = template_particle
        
        # Check if position y is within 3/4 of the search area
        indices_y = (particles[:, :, 3] > template_particle[:, 3] + 3*search_area[:, 1]/4) | (particles[:, :, 3] < template_particle[:, 3] - 3*search_area[:, 1]/4) |\
                    (particles[:, :, 3] >= frame_size[1]) | (particles[:, :, 3] < 0)
        particles[indices_y] = template_particle
        
        # Check if bbox is between the size of the search area and 16
        particles[:, :, 6] = np.minimum(search_area[:, 0], np.maximum(particles[:, :, 6], 16)) # Boxes width
        particles[:, :, 7] = np.minimum(search_area[:, 1], np.maximum(particles[:, :, 7], 16)) # Boxes height

        return particles
    
    # Measurement model using a similarity coefficient
    def measurement_model(self, coeff_sim: np.ndarray, particles: np.ndarray, template_particle:np.ndarray, R: np.ndarray) -> np.ndarray:
        # proba = ss.norm(0., R[:, 0]).pdf(coeff_sim)
        proba = 0.8*ss.norm.pdf(coeff_sim, loc=0, scale=R[:, 0]) + (1-0.8)*ss.norm.pdf(np.mean((particles - template_particle)**2, axis=(2, 1)), loc=0, scale=R[:, 1])
        # proba = np.exp(-R[:, 0] * coeff_sim**2)
        return proba