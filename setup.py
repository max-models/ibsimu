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
        res = (
            subprocess.check_output(["pkg-config"] + args, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return res.split() if res else []
    except:
        return []


def get_pkg_config_multi(flag, pkgs):
    """Query each package individually, collecting flags, silently skipping missing ones."""
    flags = []
    for pkg in pkgs:
        flags.extend(get_pkg_config([flag, pkg]))
    return flags


# IBSimu specific flags
# Since we are building in-tree, we add src/ to includes
include_dirs = ["src", Pybind11Include()]
library_dirs = ["src/.libs"]
libraries = ["ibsimu-1.0.6dev"]

# These packages are required; GTK is optional (only needed for live display)
required_pkgs = ["gsl", "libpng", "cairo", "freetype2", "fontconfig"]
optional_pkgs = ["gtk+-3.0"]

all_pkgs = required_pkgs + optional_pkgs

# Add dependencies from pkg-config (query individually so missing ones don't kill the rest)
pkg_libs = get_pkg_config_multi("--libs", all_pkgs)
for arg in pkg_libs:
    if arg.startswith("-l"):
        libraries.append(arg[2:])
    elif arg.startswith("-L"):
        library_dirs.append(arg[2:])

pkg_cflags = get_pkg_config_multi("--cflags", all_pkgs)
extra_compile_args = ["-std=c++14"]
for arg in pkg_cflags:
    if arg.startswith("-I"):
        include_dirs.append(arg[2:])
    else:
        extra_compile_args.append(arg)

# Add other libraries manually if needed (skip platform-specific ones missing on macOS)
extra_libs = ["z"]
if sys.platform != "darwin":
    extra_libs.insert(0, "rt")
libraries.extend(extra_libs)

ext_modules = [
    Extension(
        "ibsimu",
        ["python/ibsimu_py.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    name="ibsimu",
    version="1.0.6",
    author="Taneli Kalvas",
    description="Python wrapper for IBSimu",
    ext_modules=ext_modules,
    install_requires=["pybind11"],
)
