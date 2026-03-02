add_library(aksharanet_warnings INTERFACE)
target_compile_options(aksharanet_warnings INTERFACE -Wall -Wextra -Werror)

add_subdirectory(preprocessing)
add_subdirectory(detection)
add_subdirectory(recognition)
add_subdirectory(decoding)
add_subdirectory(language_model)
add_subdirectory(pipeline)

add_executable(aksharanet_cli main.cpp)
target_link_libraries(aksharanet_cli PRIVATE aksharanet_pipeline aksharanet_warnings)
