from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import os, re, eigency

def find_eigen(hint=None):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else hint
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    return None

eigen_loc = find_eigen()

setup(
    name='bcg',
    version='1.0',
    packages=['bcg'],
    url='https://github.com/jcockayne/bcg',
    license='GPL-3.0+',
    author='Jon Cockayne',
    author_email='benorn@gmail.com',
    description='Implementation of the Bayesian Conjugate Gradient Method.',
    requires=['numpy', 'scipy', 'Cython', 'eigency'],
    ext_modules=cythonize([
        Extension('*',
                  ['bcg/*.pyx'],
                  extra_objects=[],
                  extra_compile_args=["-std=c++11"],
                  extra_link_args=['cpp/lib/libbcg.a'],
                  language='c++'
                  )
        ]
    ),
    include_dirs=[numpy.get_include(), 'cpp/include'] + [eigen_loc] + eigency.get_includes(include_eigen=False)
)