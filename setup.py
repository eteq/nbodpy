from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("nbodpy.ccore", ["nbodpy/ccore.pyx"])]

setup(
  name = 'Simple Cython nbody',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)