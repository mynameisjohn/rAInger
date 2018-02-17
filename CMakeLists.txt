CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(pylLoRaWAN)

# Use the CMake find python package
SET(Python_ADDITIONAL_VERSIONS 3.4)
FIND_PACKAGE(PythonLibs)

# We use c++14 for std::integer_sequence
SET(CMAKE_CXX_FLAGS "-std=c++14 -Wall")

# Add subdir for Pyliaison (we only care about the pyl directory)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/Pyliaison/pyl)

# Use libloraGW headers
# I keep them locally but you could define this
IF(NOT DEFINED(LGW_DIR))
	SET(LGW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lora_gateway/libloragw)
ENDIF(NOT DEFINED(LGW_DIR))

# Set up include / lib directories
# The FindPythonLibs module doesn't give us the lib directory, 
# so use the library path to find it with GET_FILENAME_COMPONENT
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_INCLUDE_DIRS} ${LGW_DIR}/inc)
GET_FILENAME_COMPONENT(PYTHON_LIB_DIR ${PYTHON_LIBRARIES} DIRECTORY )

# Link directories are the python dir and the LGW dir
LINK_DIRECTORIES(${PYTHON_LIB_DIR} ${LGW_DIR})

# Make the rAInger exe, link against python, pyl, and LGW
# Add the scripts to the exe as well (so they show up in VS...)
ADD_EXECUTABLE(rAInger_pyl 
	${CMAKE_CURRENT_SOURCE_DIR}/rAInger_pyl.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/rAInger_Util.py
	${CMAKE_CURRENT_SOURCE_DIR}/rAInger.py
	)
TARGET_LINK_LIBRARIES(rAInger_pyl LINK_PUBLIC PyLiaison ${PYTHON_LIBRARIES} loragw mpsse ftdi )
