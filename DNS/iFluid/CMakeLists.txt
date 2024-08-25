file(GLOB ifluid_source "*.cpp" "*.h")
list(REMOVE_ITEM ifluid_source ${CMAKE_CURRENT_SOURCE_DIR}/iFluid.cpp)

add_library(ifluid STATIC ${ifluid_source})

target_link_libraries (ifluid frontier solver)

target_include_directories(ifluid PUBLIC 
				${frontier_include_dir}
                   		${solver_include_dir}
                              	${petsc_include_dir}
				${ifluid_include_dir}
                             	${external_include_dir})

#create excutable
add_executable(iFluid iFluid.cpp)

#include directories
target_include_directories(iFluid PUBLIC
				   ${solver_include_dir}
				   ${frontier_include_dir}	
				   ${ifluid_include_dir}
				   ${external_include_dir})

#link libraries
target_link_libraries(iFluid frontier solver ifluid ${FFTW3_LIBRARY} ${MPI_C_LIBRARIES})

