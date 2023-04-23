import numpy as np
from typing import Optional, Tuple
from .Particle import Particle, ConstAccelParticle2DBbox
from .ResampleMethods import systematic_resample
from utils.Descriptors import HOG
from utils.Similarity import Bhattacharyya_distance
from utils.Slicer import image_resize_slicing, template_image_slicing
from utils import slicer_dict

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
                slicer: str ='resize',
                descriptor=HOG,
                similarity_fn=Bhattacharyya_distance,
                resample_method_fn=systematic_resample):

        self.N = N
        self.track_dim = track_dim
        self.particle_struct = particle_struct
        self.state_dim = self.particle_struct().particle_dim

        # Set the ranges for a uniform distribution of the particles
        self.ranges = np.repeat([
                [0., init_frame.shape[1]], [0., 1.], [0., 1.],
                [0., init_frame.shape[0]], [0., 1.], [0., 1.],
                [0., init_frame.shape[1]/2], [0., init_frame.shape[0]/2]
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

        # Set the descriptor function
        self.descriptor = descriptor((self.prev_frame.shape[1], self.prev_frame.shape[0]))

        # Set the similarity measurement function
        self.similarity = similarity_fn

        # Set the slicer object
        self.slicer = slicer_dict[slicer]

        # Set the update function according to the choosen slicer
        self.update_fn = self.update_image_slicer if slicer in ['resize', 'crop'] else self.update_descriptor_slicer

        # Array of all the trackers and particles
        self.particles: np.ndarray = np.zeros((self.N, self.track_dim, self.state_dim))
        self.particles = self.particle_struct().create_gaussian_particles(self.N, self.track_dim, init_pos[:, 0], init_pos[:, 1])
        
        # Weights of each trackers
        self.weights: np.ndarray = np.ones(self.N)/self.N
        
        # Mean of all the particles for each targets
        self.mu: np.ndarray = init_pos[0, 0]

        # Standard deviation of all the particles for each targets
        self.sigma: np.ndarray = init_pos[0, 1]


        # Save the descriptor of the previous frame
        self.prev_descriptor = self.descriptor.compute(np.array([self.prev_frame]))

        # Save the descriptor of the best particle at the previous frame
        self.descriptor.update((int(self.mu[6]), int(self.mu[7])))
        template_patch = template_image_slicing(self.prev_frame, self.mu)
        self.prev_patch_descriptor = self.descriptor.compute(np.array([template_patch]))

    # Predict next state for each trackers (prior)
    def predict(self, dt: Optional[float] =1.) -> None:
        self.particles = self.particle_struct().motion_model(self.particles, self.Q_motion, dt)

    def update(self, z: np.ndarray) -> None:
        self.prev_frame = z

        self.update_fn(z)

        self.weights += 1.e-12
        self.weights /= np.sum(self.weights)

    # Update each tracker belief with observations (z).
    # With image slicing
    def update_image_slicer(self, z: np.ndarray) -> None:

        image_slice = self.slicer(self.particles, z, self.mu)

        descriptor_result = self.descriptor.compute(image_slice)

        coeff_sim = self.similarity(descriptor_result, self.prev_patch_descriptor[0])

        self.weights = self.particle_struct().measurement_model(coeff_sim, self.R)

    # Update each tracker belief with observations (z).
    # With descriptor slicing
    def update_descriptor_slicer(self, z: np.ndarray) -> None:

        descriptor_result = self.descriptor.compute(z)

        descriptor_slice = self.slicer(descriptor_result)

        coeff_sim = self.similarity(descriptor_slice, self.prev_descriptor[0])

        self.weights *= self.particle_struct().measurement_model(coeff_sim, self.R)

    # Computation of the mean and standard deviation of the particles for each targets (estimate)
    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        self.mu = np.average(self.particles, weights=self.weights, axis=0)[0]
        self.sigma = np.average((self.particles - self.mu)**2, weights=self.weights, axis=0)[0]

        template_patch = template_image_slicing(self.prev_frame, self.mu)

        self.descriptor.update((self.prev_frame.shape[1], self.prev_frame.shape[0]))
        self.prev_descriptor = self.descriptor.compute(np.array([self.prev_frame]))

        self.descriptor.update((template_patch.shape[1], template_patch.shape[0]))
        self.prev_patch_descriptor = self.descriptor.compute(np.array([template_patch]))

        return (self.mu, self.sigma)

    # Compute the effective N value
    def neff(self) -> float:
        return 1. / np.sum(np.square(self.weights))

    # Perform resample 
    def resample(self, fraction: Optional[float] =1./4.) -> None:
        if self.neff() < self.N * fraction:
            indexes = self.resample_method(self.weights)
            self.particles[:] = self.particles[indexes]
            self.weights.resize(self.N)
            self.weights.fill(1/self.N)

    # Perform one pass of the particle filter
    def forward(self, z: np.ndarray, dt: float =1., fraction: float =1./4.) -> None:
        self.predict(dt)
        self.update(z)
        self.resample(fraction)
        self.estimate()