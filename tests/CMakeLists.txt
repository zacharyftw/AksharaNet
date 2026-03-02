include(GoogleTest)

add_executable(test_stub test_stub.cpp)
target_link_libraries(test_stub PRIVATE GTest::gtest_main)
gtest_discover_tests(test_stub)
