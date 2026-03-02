include(GoogleTest)

function(add_aksharanet_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE GTest::gtest_main aksharanet_pipeline)
    gtest_discover_tests(${name})
endfunction()

add_aksharanet_test(test_preprocessing)
add_aksharanet_test(test_detection)
add_aksharanet_test(test_recognition)
add_aksharanet_test(test_decoding)
add_aksharanet_test(test_language_model)
add_aksharanet_test(test_pipeline)
