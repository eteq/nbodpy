from __future__ import division

import numpy as np
#np.seterr(invalid='ignore',divide='ignore')



#<-------------------------IC Generators--------------------------------------->

def normal_ics(nparticles,pscale=1,vscale=1,masses=None):
    """
    Generates `nparticles` particles with normally distributed locations and
    speeds.
    """
    from core import Particles
    
    pos = pscale*np.randn(3,nparticles)
    vel = vscale*np.randn(3,nparticles)
    
    if masses is None:
        return Particles(pos,vel)
    else:
        return Particles(pos,vel,masses)
    
def uniform_ics(nparticles,pscale=1,vscale=1,masses=None):
    """
    Generates `nparticles` particles with uniformly distributed locations
    (centered at the origin with box size `pscale`) and uniform velocities.
    """
    from core import Particles
    
    pos = pscale*(np.rand(3,nparticles)-.5)
    vel = vscale*(np.rand(3,nparticles)-.5)
    
    if masses is None:
        return Particles(pos,vel)
    else:
        return Particles(pos,vel,masses)
    
def uniform_normal_ics(nparticles,pscale=1,vscale=1,masses=None):
    """
    Generates `nparticles` particles with uniformly distributed locations
    (centered at the origin with box size `pscale`) and gaussian velocities.
    """
    from core import Particles
    
    pos = pscale*(np.rand(3,nparticles)-.5)
    vel = vscale*np.randn(3,nparticles)
    
    if masses is None:
        return Particles(pos,vel)
    else:
        return Particles(pos,vel,masses)
    
def keplerian_ics(nparticles,zscale=0,vzscatter=0,radialprof='exponential',
                             zprof='gaussian',centermass=1):
    """
    Sets up initial conditions corresponding to Keplerian rotation about the origin.
    
    :param zscatter: amplitude of scatter in z-direction (0 for none)
    :param vzscatter: amplitude of gaussian scatter in z-velocity
    :param radialprof: 
        Can be 'exponential','gaussian','power#.#','ring', or 'uniform'. All are 
        scaled to 1. (#.# in power gives power law index)
    :param zprof: 
        Can be 'gaussian','exponential','sechsq','power#.#'
    :param centermass: mass of central point-potential
    
    :returns: 
        2-tuple *particles,potential) - the potential is a point potential with
        the appropriate mass.
    """
    from core import Particles,PointPotential
    
    if radialprof == 'exponential':
        r = -np.log(1-np.random.rand(nparticles))
    elif radialprof == 'gaussian':
        r = np.random.randn(nparticles)
    elif radialprof == 'ring':
        r = np.ones(nparticles)
    elif radialprof == 'uniform':
        r = np.random.rand(nparticles)
    elif radialprof.startswith('power'):
        alpha = float(radialprof[5:])
        r = ((1-alpha)*np.random.rand(nparticles))**(1/(1-alpha))
    else:
        raise ValueError('Invalid radial profile')
    
    phi = 2*np.pi*np.random.rand(nparticles)
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    
    if zprof == 'exponential':
        z = -np.log(1-np.random.rand(nparticles))
    elif zprof == 'gaussian':
        z = np.random.randn(nparticles)
    elif zprof == 'sechsq':
        z = np.arctanh(np.random.rand(nparticles))
    elif zprof.startswith('power'):
        alpha = float(radialprof[5:])
        z = ((1-alpha)*np.random.rand(nparticles))**(1/(1-alpha))
    else:
        raise ValueError('Invalid z profile')
    z *= zscale
    
    vz = vzscatter*np.random.randn(nparticles)
    
    vcirc = (centermass/r)**0.5 
    vxy = vcirc - vz
    
    s = np.hypot(x,y)
    vx = y*vxy/s
    vy = -x*vxy/s
    
    ps = Particles(x,y,z,vx,vy,vz)
    pot = PointPotential(centermass)
    
    return ps,pot
    