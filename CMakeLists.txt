cmake_minimum_required(VERSION 3.15...4.0)

# Project Name
if(NOT SOQUE_PROJECT_NAME)
  set(SOQUE_PROJECT_NAME soque)
  set(SOQUE_PROJECT_VERSION 1.0)
endif(NOT SOQUE_PROJECT_NAME)

project(
    ${SOQUE_PROJECT_NAME}
    VERSION ${SOQUE_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Options
OPTION(PARALLEL_HDF5 "Whether to use Parallel HDF5" OFF)
# Additional CMake Files
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind/cmake/")
#
#
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(PROJECT_NBMOD ${PROJECT_NAME}_ext)
set(PROJECT_INSTALL_DIR ${PROJECT_NAME})
#
message(STATUS "CMake Variables/Options::")
message(STATUS " -> PROJECT NAME      :  ${PROJECT_NAME}")
message(STATUS " -> PROJECT VERSION   :  ${PROJECT_VERSION}")
message(STATUS " -> NBMODULE NAME     :  ${PROJECT_NBMOD}")
message(STATUS " -> NBMODULE INSTALL  :  ${PROJECT_INSTALL_DIR}")
message(STATUS " -> CMAKE MODULE PATH :  ${CMAKE_MODULE_PATH}")
message(STATUS " -> CMAKE VERBOSE     :  ${CMAKE_VERBOSE_MAKEFILE}")
message(STATUS " -> BUILD             :  ${CMAKE_BUILD_TYPE}")
message(STATUS " -> PARALLEL_HDF5     :  ${PARALLEL_HDF5}")

#
include_directories("${PROJECT_SOURCE_DIR}/include/")

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
# find_package(nanobind CONFIG REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  ${PROJECT_NBMOD}

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/soque_ext.cpp
)
add_compile_definitions(SOQUE_VERSION=${PROJECT_VERSION})
add_compile_definitions(SOQUE_MODULE_NAME=${PROJECT_NBMOD})

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP :")
  message(STATUS "  -> version     : ${OpenMP_CXX_VERSION}")
  #message(STATUS "  -> headers     : ${OpenMP_CXX_INCLUDE_DIRS}")
  message(STATUS "  -> libraries   : ${OpenMP_CXX_LIBRARIES}")
  include_directories(${OpenMP_CXX_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NBMOD} PUBLIC OpenMP::OpenMP_CXX)
endif()

# External header-only library: mxx
add_library(mxx INTERFACE)
target_include_directories(mxx INTERFACE
                           ${CMAKE_CURRENT_SOURCE_DIR}/ext/mxx/include)
target_link_libraries(${PROJECT_NBMOD} PUBLIC mxx)

# External header-only library: mxx
add_library(prettyp INTERFACE)
target_include_directories(prettyp INTERFACE
                           ${CMAKE_CURRENT_SOURCE_DIR}/ext/mxx/ext)
target_link_libraries(${PROJECT_NBMOD} PUBLIC prettyp)

# MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
  message(STATUS "MPI :")
  message(STATUS "  -> version     : ${MPI_VERSION}")
  message(STATUS "  -> cxx compiler: ${MPI_CXX_COMPILER}")
  message(STATUS "  -> headers     : ${MPI_INCLUDE_PATH}")
  message(STATUS "  -> flags       : ${MPI_CXX_COMPILE_DEFINITIONS}")
  message(STATUS "  -> libs        : ${MPI_LIBRARIES}")
  #include_directories(${MPI_INCLUDE_PATH})
  #add_definitions(${MPI_COMPILE_FLAGS})
  #target_link_libraries(${PROJECT_NBMOD} PUBLIC ${MPI_LIBRARIES})
  target_link_libraries(${PROJECT_NBMOD} PUBLIC MPI::MPI_CXX)
endif (MPI_FOUND)

# HDF5
if (PARALLEL_HDF5)
  set(HDF5_PREFER_PARALLEL true)
endif (PARALLEL_HDF5)
#set(HDF5_USE_STATIC_LIBRARIES TRUE)
find_package(HDF5 COMPONENTS C REQUIRED)
if (HDF5_FOUND)
  if ((NOT HDF5_IS_PARALLEL) AND PARALLEL_HDF5)
    message(WARNING "Can NOT find parallel HDF5, using serial HDF5 instead.")
  endif ((NOT HDF5_IS_PARALLEL) AND PARALLEL_HDF5)
  message(STATUS "HDF5 :")
  message(STATUS "  -> version     : ${HDF5_VERSION}")
  message(STATUS "  -> is parallel : ${HDF5_IS_PARALLEL}")
  message(STATUS "  -> headers     : ${HDF5_C_INCLUDE_DIRS}")
  message(STATUS "  -> definitions : ${HDF5_C_DEFINITIONS}")
  message(STATUS "  -> lib dirs    : ${HDF5_C_LIBRARY_DIRS}")
  message(STATUS "  -> lib names   : ${HDF5_C_LIBRARY_NAMES}")
  message(STATUS "  -> libs        : ${HDF5_C_LIBRARIES}")
  # add_definitions(${HDF5_C_DEFINITIONS})
  # include_directories(${HDF5_C_INCLUDE_DIRS})
  # target_link_directories(${PROJECT_NBMOD} PUBLIC ${HDF5_C_LIBRARY_DIRS})
  # target_link_libraries(${PROJECT_NBMOD} PUBLIC ${HDF5_C_LIBRARIES})
  target_link_libraries(${PROJECT_NBMOD} PUBLIC HDF5::HDF5)
  if (HDF5_IS_PARALLEL)
    add_compile_definitions(USE_PARALLEL_HDF5)
  endif (HDF5_IS_PARALLEL)
endif (HDF5_FOUND)

# trng4 library
set(TRNG_ENABLE_TESTS OFF)
set(TRNG_ENABLE_EXAMPLES OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/trng4)
target_link_libraries(${PROJECT_NBMOD} PUBLIC trng4)
#
get_target_property(TRNG4_VERSION trng4 VERSION)
message(STATUS "TRNG4 :")
message(STATUS "  -> version     : ${TRNG4_VERSION}" )

# Install directive for scikit-build-core
install(TARGETS ${PROJECT_NBMOD} LIBRARY DESTINATION ${PROJECT_INSTALL_DIR})
