set(PN ZeroMQ)
set(_CONDA_PREFIX "/workspaces/timesfm/venv")
set(${PN}_INCLUDE_DIR "${_CONDA_PREFIX}/include")
set(${PN}_LIBRARY "${_CONDA_PREFIX}/lib/libzmq.so")
set(${PN}_STATIC_LIBRARY "${_CONDA_PREFIX}/lib/libzmq.a")
unset(_CONDA_PREFIX)

set(${PN}_FOUND TRUE)

# add libzmq-4.3.3 cmake targets

# only define targets once
# this file can be loaded multiple times
if (TARGET libzmq)
  return()
endif()

add_library(libzmq SHARED IMPORTED)
set_property(TARGET libzmq PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${${PN}_INCLUDE_DIR}")
set_property(TARGET libzmq PROPERTY IMPORTED_LOCATION "${${PN}_LIBRARY}")

add_library(libzmq-static STATIC IMPORTED "${${PN}_INCLUDE_DIR}")
set_property(TARGET libzmq-static PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${${PN}_INCLUDE_DIR}")
set_property(TARGET libzmq-static PROPERTY IMPORTED_LOCATION "${${PN}_STATIC_LIBRARY}")

