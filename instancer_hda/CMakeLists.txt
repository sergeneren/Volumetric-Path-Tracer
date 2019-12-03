#add houdini cmake lists

set(HOUDINI_CMAKE_DIR "C:/Program Files/Side Effects Software/Houdini 17.5.327/toolkit/cmake")

list(APPEND CMAKE_PREFIX_PATH ${HOUDINI_CMAKE_DIR})

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini)

file (GLOB Headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" )
file (GLOB Sources "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )

add_library(instancer_hda SHARED ${Headers} ${Sources})
target_link_libraries(instancer_hda Houdini)

message(STATUS "Houdini directory: ${Houdini_DIR}")

houdini_configure_target( instancer_hda )


#copy icon to %HOME%/%HOUDINI_VERSION%/config/icons folder

set(TARGET_FILE "${${CMAKE_CURRENT_SOURCE_DIR}}/icon/SOP_VPT_instance.svg")
set(TARGET_DIR "")

houdini_get_default_install_dir(TARGET_DIR)

set(TARGET_DIR "${TARGET_DIR}\\config\\icons")

add_custom_command(TARGET instancer_hda
					POST_BUILD
					COMMAND ${CMAKE_COMMAND} -E copy "${TARGET_FILE}" "${TARGET_DIR}"
					COMMENT "Copying ${TARGET_FILE} to ${TARGET_DIR}")

message(STATUS "${TARGET_FILE} will be copied to ${TARGET_DIR}")