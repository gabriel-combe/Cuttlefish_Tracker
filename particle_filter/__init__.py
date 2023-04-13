from .ParticleFilter import ParticleFilter
from .Particle import ConstAccelParticle2DFixBbox, ConstAccelParticle2DBbox
from .ResampleMethods import systematic_resample, residual_resample, stratified_resample, multinomial_resample

particle_dict = {
    'cap2Dfbb': ConstAccelParticle2DFixBbox,
    'cap2Dbb': ConstAccelParticle2DBbox
    }

resample_dict = {
    'systematic': systematic_resample,
    'residual': residual_resample,
    'stratified': stratified_resample,
    'multinomial': multinomial_resample
    }