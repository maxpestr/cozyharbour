cmake_minimum_required(VERSION 3.15)
project(core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Пусть CMake сам найдёт Python из твоей Anaconda
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}" "${Python3_SITEARCH}" )

# Если не найден — ищем через pip (fallback)
if (NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Добавляем модуль
pybind11_add_module(core core.cpp)
