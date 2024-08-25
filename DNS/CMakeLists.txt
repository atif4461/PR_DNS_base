cmake_minimum_required (VERSION 3.20.0)
project (climate C CXX CUDA)

set (CMAKE_C_FLAGS "-Wno-implicit-function-declaration")
set (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fopenmp -D__HDF5__ ") 
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D__HDF5__ ") 
##set (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -D__NO_RND__")      # uncomment to disable random numbers for reproducibility
##set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__NO_RND__")      # uncomment to disable random numbers for reproducibility
set (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -D__PRDNS_TIMER__") # uncomment to enable detailed timings
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__PRDNS_TIMER__") # uncomment to enable detailed timings
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ljpeg")

#set (MPI_HOME "/home/atif/packages/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-11.4.0/mpich-4.2.1-adou5vzody4nf4iagajbb7oxektbnczz/") # dahlia
set (MPI_HOME "/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/") # perlmutter
set (CMAKE_CUDA_FLAGS "-I/${MPI_HOME}/include/ -L/${MPI_HOME}/lib -lmpi -O3")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -rdynamic -lineinfo -D__CUDA__ -O3 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86")

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g ")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g ")

#detect dependencies
list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

find_package (MPI REQUIRED)

find_package (HDF4 QUIET)

find_package (GD QUIET)

set (heFFTe_PATH "/global/homes/a/atif/packages/heffte-2.4.0/")
find_package(Heffte 2.4.0 REQUIRED PATHS ${heFFTe_PATH})

if (MPI_FOUND) 
    MESSAGE(STATUS "Found MPI: ${MPI_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__MPI__")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__MPI__")
else ()
    MESSAGE(FATAL_ERROR "Could NOT find MPI")
endif()

if (HDF4_FOUND)
    MESSAGE(STATUS "Found HDF4: ${HDF4_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_HDF")
else ()
    MESSAGE(WARNING "HDF4 cannot be found. Set HDF4_ROOT is hdf animation is needed. Build continues ...")
endif()

if (GD_FOUND)
    MESSAGE(STATUS "Found GD: ${GD_LIBRARY}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__GD__")
else ()
    MESSAGE(WARNING "GD cannot be found. Set GD_DIR if gd animation is needed. Build continues ...")
endif()

if (Heffte_FFTW_FOUND)
    MESSAGE(STATUS "Found heFFTe ${heFFTe_PATH}")
else ()
    MESSAGE(WARNING "heFFTe cannot be found. Build continues ...")
endif()

#dependencies detection finished

set(frontier_include_dir ${CMAKE_SOURCE_DIR}/src/front
                         ${CMAKE_SOURCE_DIR}/src/intfc
                         ${CMAKE_SOURCE_DIR}/src/util
                         ${CMAKE_SOURCE_DIR}/src)

set(solver_include_dir ${CMAKE_SOURCE_DIR}/solver)

set(ifluid_include_dir ${CMAKE_SOURCE_DIR}/iFluid)

set(climate_include_dir ${CMAKE_SOURCE_DIR}/climate)

set(project_include_dir ${frontier_include_dir}
			${solver_include_dir}
			${ifluid_include_dir}
			${climate_include_dir})

#external libraries include directories
set(external_include_dir 
		${MPI_C_INCLUDE_PATH})

#external libraries link dir
set(external_lib_dir 
                ${MPI_C_LIBRARIES}) 

if (GD_FOUND)
	list( APPEND external_include_dir ${GD_INCLUDE_DIR})
	list( APPEND external_lib_dir ${GD_LIBRARY})
endif()

if (HDF4_FOUND)
	list( APPEND external_include_dir ${HDF4_INCLUDE_DIR})
	list( APPEND external_lib_dir ${HDF4_LIBRARIES})
endif()

if (Heffte_FFTW_FOUND)
	list( APPEND external_include_dir "${heFFTe_PATH}/include")
	list( APPEND external_lib_dir "${heFFTe_PATH}/lib")
endif()

# Recurse into the following subdirectories
add_subdirectory (climate)
add_subdirectory (iFluid)
add_subdirectory (src)
add_subdirectory (solver)
