import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class Pybind11Include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()

def get_pkg_config(args):
    try:
        res = subprocess.check_output(['pkg-config'] + args).decode('utf-8').strip()
        return res.split() if res else []
    except:
        return []

# IBSimu specific flags
# Since we are building in-tree, we add src/ to includes
include_dirs = ['src', Pybind11Include()]
library_dirs = ['src/.libs']
libraries = ['ibsimu-1.0.6dev']

# Add dependencies from pkg-config
pkg_libs = get_pkg_config(['--libs', 'gtk+-3.0', 'gsl', 'libpng', 'cairo', 'freetype2', 'fontconfig'])
# Extract library names from -l flags
for arg in pkg_libs:
    if arg.startswith('-l'):
        libraries.append(arg[2:])
    elif arg.startswith('-L'):
        library_dirs.append(arg[2:])

pkg_cflags = get_pkg_config(['--cflags', 'gtk+-3.0', 'gsl', 'libpng', 'cairo', 'freetype2', 'fontconfig'])
extra_compile_args = ['-std=c++11']
for arg in pkg_cflags:
    if arg.startswith('-I'):
        include_dirs.append(arg[2:])
    else:
        extra_compile_args.append(arg)

# Add other libraries manually if needed
libraries.extend(['rt', 'umfpack', 'amd', 'blas', 'z'])

ext_modules = [
    Extension(
        'ibsimu',
        ['python/ibsimu_py.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
]

setup(
    name='ibsimu',
    version='1.0.6',
    author='Taneli Kalvas',
    description='Python wrapper for IBSimu',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
)
