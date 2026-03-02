include(FetchContent)

if(AKSHARANET_SYSTEM_KENLM)
    find_package(kenlm REQUIRED)
else()
    FetchContent_Declare(
        kenlm
        GIT_REPOSITORY https://github.com/kpu/kenlm.git
        GIT_TAG        master
    )
    set(COMPILE_TESTS     OFF CACHE BOOL "" FORCE)
    set(COMPILE_BENCHMARK OFF CACHE BOOL "" FORCE)

    # Populate without adding subdirectory, then add with EXCLUDE_FROM_ALL
    # so only targets we explicitly link (kenlm, kenlm_util) get built —
    # KenLM's executables (filter, build_binary, etc.) need Boost::thread
    # which is header-only in Conan and missing the compiled runtime.
    FetchContent_GetProperties(kenlm)
    if(NOT kenlm_POPULATED)
        FetchContent_Populate(kenlm)
        add_subdirectory(${kenlm_SOURCE_DIR} ${kenlm_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    # Restore C++20 after KenLM overrides it to C++11
    set(CMAKE_CXX_STANDARD 20 CACHE STRING "" FORCE)
endif()
