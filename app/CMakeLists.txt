
cmake_minimum_required(VERSION 3.16)
set (CMAKE_Fortran_COMPILER "gfortran.exe")

project(pll_solver VERSION 1.2.0 LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES 
    pll/helix_func.f90
    pll/matrix_mod.f90
    pll/points_mod.f90
    pll/propeller.f90
    pll/operating.f90
    json-lib/*.f90
)

add_library(pll SHARED pll/pll.f90 ${SOURCES})
set_target_properties(pll PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/mod"
)

add_executable(propeller_lifting_line pll/propeller_lifting_line.f90 ${SOURCES})

find_path(MKL_INCLUDE_DIR mkl.h HINTS ENV MKLROOT PATH_SUFFIXES include)
find_library(MKL_LIBRARY mkl_rt HINTS ENV MKLROOT PATH_SUFFIXES lib lib/intel64)

if(MKL_INCLUDE_DIR AND MKL_LIBRARY)
    message(STATUS "Found MKL: ${MKL_LIBRARY}")

    include_directories(${MKL_INCLUDE_DIR})

    target_link_libraries(pll PRIVATE ${MKL_LIBRARY})
    
    #target_link_options(pll PRIVATE "-Wl,--start-group ${MKL_LIBRARY} -Wl,--end-group")

    target_link_libraries(propeller_lifting_line PRIVATE ${MKL_LIBRARY})

    #target_link_options(propeller_lifting_line PRIVATE "-Wl,--start-group ${MKL_LIBRARY} -Wl,--end-group")
else()
    message(FATAL_ERROR "MKL not found. Please ensure MKL is installed and MKLROOT is set.")
endif()




