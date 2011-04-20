from __future__ import division

import numpy as np
#np.seterr(invalid='ignore',divide='ignore')
import abc

#<----------------------------Base classes------------------------------------->

class Integrator(object):
    """
    Integrates Hamilton's equations of motions for a given set of particles and
    a potential.  Abstract class for which :meth:`integrate` must be overridden
    """
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,particles,potential,deltat=1,orbititers=1):
        self.particles = particles
        self.potential = NullPotential() if potential is None else potential
        self.deltat = deltat
        
        self.orbititers = 0 if orbititers is None else int(orbititers)
        self.t = 0
        
        self.clearOrbits() #also initializes
        
        if self.orbititers:
            self._orbt.append(self.t)
            self._orbits.append(self.particles.position.copy())
    
    def getOrbits(self):
        """
        Get the orbits (positions as a function of time).
        
        :returns: 
            (t,orbits) where `orbits` is an array of shape (norbits,3_ and `t` is
            an array matching the first dimension of `orbits`.
        """
        return np.array(self._orbt),np.array(self._orbits)
    
    def clearOrbits(self):
        """
        Empties the orbits.
        """
        self._orbt = []
        self._orbits = []
    
    def __call__(self,orbits=None):
        if self.orbititers==0:
            orbititers = 1
        else:
            orbititers = self.orbititers
        
        for i in range(orbits):
            self.integrate(orbititers)
            self._orbt.append(self.t)
            self._orbits.append(self.particles.position.copy())
        
    @abc.abstractmethod
    def integrate(self,iters):
        """
        Perform the actual integration using the particles and potential. Should
        be overridden in subclasses. :attr:`deltat` must be updated here when 
        iteration is finished.
        
        .. note:: 
            Orbits will *not* be saved if this is called directly - the better
            way to integrate is to call the :class:`Integrator` object itself.
        
        :param int iters:  The number of iterations to perform
        """
        raise NotImplementedError
    
    def plot_orbits(self,xaxis='x',yaxis='y',clf=True,**kwargs):
        """
        Plots a 2D plot of the particles and their orbits.
        
        :param str xaxis: The axis to plot on the plot x-axis - 'x','y', or 'z'
        :param str yaxis: The axis to plot on the plot y-axis - 'x','y', or 'z'
        :param bool clf: If True, clear figure before plotting.
        
        kwargs are passed into :func:`matplotlib.pyplot.plot`.
        
        :returns: 
            the result of :func:`matplotlib.pyplot.plot` and the result of
            :func:`matplotlib.pyplot.quiver`
        """
        from matplotlib import pyplot as plt
        
        imap = {'x':0,'y':1,'z':2}
        
        if xaxis not in imap:
            raise ValueError('invalid xaxis %s'%xaxis)
        if yaxis not in imap:
            raise ValueError('invalid yaxis %s'%yaxis)
        
        ix = imap[xaxis]
        iy = imap[yaxis]
        
        x0 = self.particles._pos[:,ix]
        vx0 = self.particles._vel[:,ix]
        y0 = self.particles._pos[:,iy]
        vy0 = self.particles._vel[:,iy]
        
        orbt,orbs = self.getOrbits()
        ox = orbs[:,:,ix]
        oy = orbs[:,:,iy]
        
        if clf:
            plt.clf()
            
        pltres = plt.plot(ox,oy,**kwargs)
            
        quiverres = plt.quiver(x0,y0,vx0,vy0)
            
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        
        return pltres,quiverres

    
class Particles(object):
    """
    A collection of particles with positions and instantaneous velocities.
    """
    
    def __init__(self,*args):
        """
        Particles can be initialized as:
        
        * Particles(n)
          Generates n particles with initial positions and velocities set to 0
          and masses set to 1.
          
        * Particles(pos,vel)
          `pos` and `vel` are both 3 x N arrays (x,y,z) and (vx,vy,vz),
          respectively. Masses all set to 1.
        
        * Particles(pos,vel,m)
          `pos` and `vel` are both 3 x N arrays (x,y,z) and (vx,vy,vz),
          respectively. `m` is a length N array of masses.
          
        * Particles(x,y,z,vx,vy,vz)
          `x` `y` and `z` are length N arrays with the positions, and `vx`,`vy`,
          and `vz` are length N arrays with velocities.  Masses all set to `.
          
        * Particles(x,y,z,vx,vy,vz,m)
          `x` `y` and `z` are length N arrays with the positions, and `vx`,`vy`,
          and `vz` are length N arrays with velocities.  `m` is a length N array
          of masses.
          
        :raises TypeError: If an invalid number of arguments are given.
        :raises ValueError: If any dimensions are invalid
        
        """
        if len(args)==1:
            n = args[0]
            pos = np.zeros((3,n))
            vel = np.zeros((3,n))
            mass = np.ones(n)
            
        elif len(args)==2 or len(args)==3:
            pos = np.transpose(args[0]).copy()
            vel = np.transpose(args[1]).copy()
            if len(args)==3:
                mass = args[2].copy()
            else:
                mass = np.ones(pos.shape[0])
                
        elif len(args)==6 or len(args)==7:
            pos = np.transpose([args[0],args[1],args[2]])
            vel = np.transpose([args[4],args[5],args[6]])
            if len(args)==7:
                mass = args[6].copy()
            else:
                mass = np.ones(pos.shape[0])
                
        else:
            raise TypeError('Must provide 1,2,3,6,or 7 arguments to Particles initializer (%i given)'%len(args))
        
        if len(pos.shape)!=2:
            raise ValueError('Positions are not 2d arrays')
        if len(vel.shape)!=2:
            raise ValueError('Velocities are not 2d arrays')
        if len(mass.shape)!=1:
            raise ValueError('Masses are not 1d arrays')
        
        if not pos.shape[1] == 3:
            raise ValueError('Positions are not 3-dimensional (nd=%i)'%pos.shape[0])
        if not vel.shape[1] == 3:
            raise ValueError('Positions are not 3-dimensional (nd=%i)'%vel.shape[0])
            
        if not (pos.shape[0] == vel.shape[0] == mass.size):
            raise ValueError('n does not match for pos,vel, and mass: %i,%i,%i'%(pos.shape[1],vel.shape[1],mass.size))
        
        
            
        self._pos = pos
        self._vel = vel
        self._mass = mass
            
    @property
    def n(self):
        """
        Number of particles.
        """
        return self.mass.size
    
    @property
    def position(self):
        """
        Returns positions of particles as a N x 3 array.  Array is a view and
        can be modified in-place to change particle positions.
        """
        return self._pos
    
    @property
    def velocity(self):
        """
        Returns velocities of particles as a N x 3 array.  Array is a view and
        can be modified in-place to change particle velocities.
        """
        return self._vel
    
    @property
    def mass(self):
        """
        Returns masses of particles as a length-N array.  Array is a view and
        can be modified in-place to change particle masses.
        """
        return self._mass


    def plot_particles(self,xaxis='x',yaxis='y',scatter=True,clf=True,**kwargs):
        """
        Plots the particle locations in a 2d projection with matplotlib.
        
        :param str xaxis: The axis to plot on the plot x-axis - 'x','y', or 'z'
        :param str yaxis: The axis to plot on the plot y-axis - 'x','y', or 'z'
        :param bool scatter:
            If True, :func:`matplotlib.pyplot.scatter` will be used to plot the
            particles. Otherwise, :func:`matplotlib.pyplot.plot`, and if not
            False or None, it will be used as the format string for
            :func:`matplotlib.pyplot.plot`.
        :param bool clf: If True, clear figure before plotting.
            
        kwargs are passed into the plotting function
        
        :returns: the result of the plotting command
        """
        from matplotlib import pyplot as plt
        
        imap = {'x':0,'y':1,'z':2}
        
        if xaxis not in imap:
            raise ValueError('invalid xaxis %s'%xaxis)
        if yaxis not in imap:
            raise ValueError('invalid yaxis %s'%yaxis)
        
        x = self._pos[:,imap[xaxis]]
        y = self._pos[:,imap[yaxis]]
        
        if clf:
            plt.clf()
        
        if scatter is True:
            res = plt.scatter(x,y,**kwargs)
        elif scatter is False:
            res = plt.plot(x,y,**kwargs)
        else:
            res = plt.plot(x,y,scatter,**kwargs)
            
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
            
        return res
        
    def plot_velocities(self,xaxis='x',yaxis='y',clf=True,**kwargs):
        """
        Plots a 2D quiver plot of the particles and their velocities with
        matplotlib.
        
        :param str xaxis: The axis to plot on the plot x-axis - 'x','y', or 'z'
        :param str yaxis: The axis to plot on the plot y-axis - 'x','y', or 'z'
        :param bool clf: If True, clear figure before plotting.
        
        kwargs are passed into :func:`matplotlib.pyplot.quiver`.
        
        :returns: the result of :func:`matplotlib.pyplot.quiver`
        """
        from matplotlib import pyplot as plt
        
        imap = {'x':0,'y':1,'z':2}
        
        if xaxis not in imap:
            raise ValueError('invalid xaxis %s'%xaxis)
        if yaxis not in imap:
            raise ValueError('invalid yaxis %s'%yaxis)
        
        ix = imap[xaxis]
        iy = imap[yaxis]
        
        x = self._pos[:,ix]
        vx = self._vel[:,ix]
        y = self._pos[:,iy]
        vy = self._vel[:,iy]
        
        if clf:
            plt.clf()
            
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        return plt.quiver(x,y,vx,vy,**kwargs)
    
    
        
class Potential(object):
    """
    A potential or potential-like construct that gives the force at a given
    particle position. Abstract base class that should implement :meth:`forces`.
    """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def forces(self,particles):
        """
        This method should be overridden to compute the specific force (e.g.
        F/mparticle) at the location of each of the supplied particles.
        
        :returns: 3-tuple (dvx/dt,dvy/dt,dvz/dt)
        """
        raise NotImplementedError
    
    
#<-------------------------Implementations------------------------------------->

class LeapfrogIntegrator(Integrator):
    """
    A drift-kick-drift or Verlet symplectic integrator.
    """
    def integrate(self,iters):
        ps = self.particles.position
        vs = self.particles.velocity
        dt = self.deltat
        dth = dt/2.
        for i in range(iters):
            #q12 = q0 + h*v/2
            #v1 = v0 + h*f(q12)
            #q1 = q12 + h*v2/2

            ps += dth*vs#->x_1/2
            
            fs = self.potential.forces(self.particles)
            vs += dt*fs
            
            ps += dth*vs
            
            self.t += dt
    
class PointPotential(Potential):
    """
    Potential of a point mass at a given position. Mass units assume G=1.
    """
    
    def __init__(self,mass,position=(0,0,0)):
        self.mass = mass
        self.position = np.array(position)
        
    def forces(self,particles):
        posoff = particles.pos - self.position
        rsq = np.sum(posoff**2,axis=1)
        
        return self.mass*posoff-rsq**-1.5
    
class NullPotential(Potential):
    """
    Potential of nothing (e.g. zero force)
    """
    def forces(self,particles):
        return np.zeros((particles.n,3))
    
class CompositePotential(Potential):
    """
    Potential computed by summing the effects of a set of other potentials.
    """
    
    def __init__(self,potentials):
        self.potentials = potentials
        
    def forces(self,particles):
        return np.sum([p.forces(particles) for p in self.potentials],axis=0)
    
#<-------------------------IC Generators--------------------------------------->

def normal_ics(nparticles,pscale=1,vscale=1,masses=None):
    """
    Generates `nparticles` particles with normally distributed locations and
    speeds.
    """
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
    pos = pscale*(np.rand(3,nparticles)-.5)
    vel = vscale*np.randn(3,nparticles)
    
    if masses is None:
        return Particles(pos,vel)
    else:
        return Particles(pos,vel,masses)