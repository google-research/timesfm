# tdbcConfig.sh --
#
# This shell script (for sh) is generated automatically by TDBC's configure
# script. It will create shell variables for most of the configuration options
# discovered by the configure script. This script is intended to be included
# by the configure scripts for TDBC extensions so that they don't have to
# figure this all out for themselves.
#
# The information in this file is specific to a single platform.
#
# RCS: @(#) $Id$

# TDBC's version number
tdbc_VERSION=1.1.5
TDBC_VERSION=1.1.5

# Name of the TDBC library - may be either a static or shared library
tdbc_LIB_FILE=libtdbc1.1.5.so
TDBC_LIB_FILE=libtdbc1.1.5.so

# String to pass to the linker to pick up the TDBC library from its build dir
tdbc_BUILD_LIB_SPEC="-L/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/unix/pkgs/tdbc1.1.5 -ltdbc1.1.5"
TDBC_BUILD_LIB_SPEC="-L/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/unix/pkgs/tdbc1.1.5 -ltdbc1.1.5"

# String to pass to the linker to pick up the TDBC library from its installed
# dir.
tdbc_LIB_SPEC="-L/workspaces/timesfm/venv/lib/tdbc1.1.5 -ltdbc1.1.5"
TDBC_LIB_SPEC="-L/workspaces/timesfm/venv/lib/tdbc1.1.5 -ltdbc1.1.5"

# Name of the TBDC stub library
tdbc_STUB_LIB_FILE="libtdbcstub1.1.5.a"
TDBC_STUB_LIB_FILE="libtdbcstub1.1.5.a"

# String to pass to the linker to pick up the TDBC stub library from its
# build directory
tdbc_BUILD_STUB_LIB_SPEC="-L/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/unix/pkgs/tdbc1.1.5 -ltdbcstub1.1.5"
TDBC_BUILD_STUB_LIB_SPEC="-L/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/unix/pkgs/tdbc1.1.5 -ltdbcstub1.1.5"

# String to pass to the linker to pick up the TDBC stub library from its
# installed directory
tdbc_STUB_LIB_SPEC="-L/workspaces/timesfm/venv/lib/tdbc1.1.5 -ltdbcstub1.1.5"
TDBC_STUB_LIB_SPEC="-L/workspaces/timesfm/venv/lib/tdbc1.1.5 -ltdbcstub1.1.5"

# Path name of the TDBC stub library in its build directory
tdbc_BUILD_STUB_LIB_PATH="/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/unix/pkgs/tdbc1.1.5/libtdbcstub1.1.5.a"
TDBC_BUILD_STUB_LIB_PATH="/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/unix/pkgs/tdbc1.1.5/libtdbcstub1.1.5.a"

# Path name of the TDBC stub library in its installed directory
tdbc_STUB_LIB_PATH="/workspaces/timesfm/venv/lib/tdbc1.1.5/libtdbcstub1.1.5.a"
TDBC_STUB_LIB_PATH="/workspaces/timesfm/venv/lib/tdbc1.1.5/libtdbcstub1.1.5.a"

# Location of the top-level source directories from which TDBC was built.
# This is the directory that contains doc/, generic/ and so on.  If TDBC
# was compiled in a directory other than the source directory, this still
# points to the location of the sources, not the location where TDBC was
# compiled.
tdbc_SRC_DIR="/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/pkgs/tdbc1.1.5"
TDBC_SRC_DIR="/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/pkgs/tdbc1.1.5"

# String to pass to the compiler so that an extension can find installed TDBC
# headers
tdbc_INCLUDE_SPEC="-I/workspaces/timesfm/venv/include"
TDBC_INCLUDE_SPEC="-I/workspaces/timesfm/venv/include"

# String to pass to the compiler so that an extension can find TDBC headers
# in the source directory
tdbc_BUILD_INCLUDE_SPEC="-I/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/pkgs/tdbc1.1.5/generic"
TDBC_BUILD_INCLUDE_SPEC="-I/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/pkgs/tdbc1.1.5/generic"

# Path name where .tcl files in the tdbc package appear at run time.
tdbc_LIBRARY_PATH="/workspaces/timesfm/venv/lib/tdbc1.1.5"
TDBC_LIBRARY_PATH="/workspaces/timesfm/venv/lib/tdbc1.1.5"

# Path name where .tcl files in the tdbc package appear at build time.
tdbc_BUILD_LIBRARY_PATH="/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/pkgs/tdbc1.1.5/library"
TDBC_BUILD_LIBRARY_PATH="/home/conda/feedstock_root/build_artifacts/tk_1699201809505/work/tcl8.6.13/pkgs/tdbc1.1.5/library"

# Additional flags that must be passed to the C compiler to use tdbc
tdbc_CFLAGS=
TDBC_CFLAGS=

