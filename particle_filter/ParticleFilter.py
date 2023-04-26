import numpy as np
from typing import Optional, Tuple
from .Particle import Particle, ConstAccelParticle2DBbox
from .ResampleMethods import systematic_resample
from utils.Descriptors import HOG
from utils.Slicer import image_resize_slicing
from utils.Similarity import Bhattacharyya_distance

import cv2

# Class for the particle filter object
class ParticleFilter(object):
    def __init__(self, 
                N: int, 
                particle_struct: Particle =ConstAccelParticle2DBbox, 
                track_dim: int =1,
                init_pos: np.ndarray =None,
                init_frame: np.ndarray =None,
                Q_motion: Optional[np.ndarray] =None,
                R: Optional[np.ndarray] =None,
                slicer_fn=image_resize_slicing,
                descriptor=HOG,
                similarity_fn=Bhattacharyya_distance,
                resample_method_fn=systematic_resample,
                seed: int =None):

        self.N = N
        self.seed = seed
        self.track_dim = track_dim
        self.rng = np.random.default_rng(self.seed)
        self.particle_struct = particle_struct(self.rng)
        self.state_dim = self.particle_struct.particle_dim
        self.frame_size = (init_frame.shape[1], init_frame.shape[0])

        # Set the ranges for a uniform distribution of the particles
        self.ranges = np.repeat([
                [0, init_frame.shape[1]], [0, 1], [0, 1],
                [0, init_frame.shape[0]], [0, 1], [0, 1],
                [0, init_frame.shape[1]//2], [0, init_frame.shape[0]//2]
            ], [self.track_dim], axis=0)
        
        # If we only give an array we repeat it for all targets
        self.init_pos = init_pos
        if self.init_pos is not None and np.isscalar(self.init_pos[0, 0]):
            self.init_pos = np.repeat([self.init_pos], self.track_dim, axis=0)

        # If we don't give Q_motion, we use defaults standard deviation.
        # If we only give a scalar we repeat it for all targets
        self.Q_motion = Q_motion
        if self.Q_motion is None:
            self.Q_motion = np.ones((self.track_dim, self.state_dim))
        elif np.isscalar(self.Q_motion):
            self.Q_motion = np.ones((self.track_dim, self.state_dim)) * self.Q_motion
        
        # If we don't give R, we use defaults standard deviation.
        # If we only give a scalar we repeat it for all targets
        self.R = R
        if self.R is None:
            self.R = np.ones((self.track_dim, self.state_dim))
        elif np.isscalar(self.R):
            self.R = np.ones((self.track_dim, self.state_dim)) * self.R

        # Save previous frame to compute the descriptor of the best particle
        self.prev_frame = init_frame
        
        # Set the resampling method function
        self.resample_method = resample_method_fn

        # Array of all the trackers and particles
        self.particles: np.ndarray = np.zeros((self.N, self.track_dim, self.state_dim))
        self.particles = self.particle_struct.create_gaussian_particles(self.N, self.track_dim, init_pos[:, 0], init_pos[:, 1])
        self.prev_particles = np.copy(self.particles)
        
        # Weights of each trackers
        self.weights: np.ndarray = np.ones(self.N)/self.N
        
        # Mean of all the particles for each targets
        self.mu: np.ndarray = init_pos[:, 0]

        # Standard deviation of all the particles for each targets
        self.sigma: np.ndarray = init_pos[:, 1]

        # Set the descriptor function
        self.descriptor = descriptor

        # Set the similarity measurement function
        self.similarity = similarity_fn

        # Set the slicer object
        self.slicer = slicer_fn

        # Save the descriptor of the best particle at the previous frame
        self.template_patch = self.slicer(np.array([self.mu]), self.prev_frame, self.mu)
        self.prev_patch_descriptor = self.descriptor.compute(self.template_patch)

        # Save previous frame to compute the descriptor of the best particle
        self.prev_frame = None

        # Save the descriptor of the best particle at the previous frame
        self.prev_descriptor = None

    # Predict next state for each trackers (prior)
    def predict(self, dt: Optional[float] =1.) -> None:
        aux_particles = np.copy(self.particles)
        self.particles = self.particle_struct.motion_model(self.particles, self.Q_motion, self.prev_particles, self.frame_size, dt)
        self.prev_particles = aux_particles

    # Update each particle's weight using a descriptor and a similarity coefficient
    def update(self, z: np.ndarray) -> None:

        self.prev_frame = z

        image_slice = self.slicer(self.particles, self.prev_frame, self.mu)

        descriptor_result = self.descriptor.compute(image_slice)

        coeff_sim = self.similarity(descriptor_result, self.prev_patch_descriptor)
        
        self.weights = self.particle_struct.measurement_model(coeff_sim, self.R)
        # self.weights = coeff

        self.weights += 1.e-12
        self.weights /= np.sum(self.weights)
        # self.weights /= np.sum(np.exp(coeff_sim))

        cv2.imshow('best particle', image_slice[np.argmax(self.weights)])
        cv2.imshow('template particle', self.template_patch[0])

    # Computation of the mean and standard deviation of the particles for each targets (estimate)
    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        self.mu = np.average(self.particles, weights=self.weights, axis=0)
        self.sigma = np.average((self.particles - self.mu)**2, weights=self.weights, axis=0)

        # Compute the descriptor of the estimate particle
        self.template_patch = self.slicer(np.array([self.mu]), self.prev_frame, self.mu)
        self.descriptor.update((int(self.mu[0, 6]), int(self.mu[0, 7])))
        self.prev_patch_descriptor = self.descriptor.compute(self.template_patch)

        return (self.mu, self.sigma)

    # Compute the effective N value
    def neff(self) -> float:
        return 1. / np.sum(np.square(self.weights))

    # Perform resample 
    def resample(self, fraction: Optional[float] =1./4.) -> None:
        print(self.neff())
        if self.neff() < self.N * fraction:
            indexes = self.resample_method(self.weights)
            print(indexes)
            self.particles[:] = self.particles[indexes]
            self.weights.resize(self.N)
            self.weights.fill(1/self.N)

    # Perform one pass of the particle filter
    def forward(self, z: np.ndarray, dt: float =1., fraction: float =1./4.) -> None:
        self.predict(dt)
        self.update(z)
        self.resample(fraction)
        self.estimate()