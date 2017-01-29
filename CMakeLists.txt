cmake_minimum_required(VERSION 3.1)

project(FlagSim)
add_executable(${PROJECT_NAME} "Flagsim.c")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(SDL REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(FlagSim Threads::Threads ${SDL_LIBRARY} ${OPENGL_gl_LIBRARY} m)

set(CMAKE_C_FLAGS "-Wall")
