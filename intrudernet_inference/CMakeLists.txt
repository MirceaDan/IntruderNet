cmake_minimum_required(VERSION 3.16)
project(intrudernet_inference LANGUAGES CXX)

# =========================
# Compiler settings
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================
# OpenCV
# =========================
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# =========================
# LibTorch
# =========================
set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "Found Torch")

# =========================
# Executable
# =========================
add_executable(intrudernet
    src/main.cpp
)

# =========================
# Include paths
# =========================
target_include_directories(intrudernet PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# =========================
# Link libraries
# =========================
target_link_libraries(intrudernet
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# =========================
# Torch compile flags
# =========================
set_property(TARGET intrudernet PROPERTY CXX_STANDARD 17)

# Fix for Raspberry Pi + GCC
target_compile_definitions(intrudernet PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
)

# =========================
# RPATH (important on Pi)
# =========================
set_target_properties(intrudernet PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
