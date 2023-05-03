from .ParticleFilter import ParticleFilter
from .Particle import ConstAccelParticle2DFixBbox, ConstAccelParticle2DBbox, PredPosParticle2DBbox
from .ResampleMethods import Systematic, Residual, Stratified, Multinomial

particle_dict = {
    'cap2Dfbb' : ConstAccelParticle2DFixBbox,
    'cap2Dbb' : ConstAccelParticle2DBbox,
    'ppp2Dbb' : PredPosParticle2DBbox
}

resample_dict = {
    'systematic': Systematic,
    'residual': Residual,
    'stratified': Stratified,
    'multinomial': Multinomial
}