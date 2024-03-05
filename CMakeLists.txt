cmake_minimum_required(VERSION 3.5)

# Project
project(QLN)

# Sources
file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS include/*.hpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Includes
target_include_directories(${PROJECT_NAME} PUBLIC include)

# Flags
target_compile_options(${PROJECT_NAME} PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -O3 -Wall -Wextra -Werror -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wconversion -Wunreachable-code -Wno-error=maybe-uninitialized -Wpedantic -Wno-error=float-equal>
)

# Download/Update librairies if needed
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodules update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --remote --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --remote --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/PNEATM")	# nothing has been created
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Add Libraries

set(BUILD_EXAMPLES OFF)
add_subdirectory(lib/PNEATM)

set(BUILD_SHARED_LIBS OFF)
set(BUILD_PROGRAMS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_REGTEST OFF)
set(ENABLE_PACKAGE_CONFIG OFF)
set(INSTALL_PKGCONFIG_MODULE OFF)
set(INSTALL_MANPAGES OFF)
set(BUILD_TESTING OFF)
set(ENABLE_CPACK OFF)
add_subdirectory(lib/sndfile)

set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTS OFF)
set(ENABLE_THREADS ON)
set(WITH_COMBINED_THREADS ON)
set(ENABLE_AVX2 ON)
add_subdirectory(lib/fftw)
target_include_directories(${PROJECT_NAME} PUBLIC lib/fftw/api)

# Link Libraries
target_link_libraries(${PROJECT_NAME} PRIVATE pneatm sndfile fftw3)
