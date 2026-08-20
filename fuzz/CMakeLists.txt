# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The coverage guided target of Section 17.3, built only when
# -DNPU_ENABLE_FUZZERS=ON. The top level CMakeLists carries the option and
# prints an off line when it is not set, because a step that is off says so.
#
# -fsanitize=fuzzer supplies main(), so this is an executable with no main of
# its own. The address and undefined sanitizers are not added here: they belong
# to the whole build, because a fuzz target instrumented against a library that
# is not tells you only about the target. The sanitizers job of Section 19
# configures the build directory with them and this file inherits them.

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "NPU_ENABLE_FUZZERS is ON but the compiler is ${CMAKE_CXX_COMPILER_ID}. "
    "-fsanitize=fuzzer is a clang feature. Configure a separate build "
    "directory with -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++.")
endif()

set(LLVM_LINK_COMPONENTS
        Support
        )

add_llvm_executable(nbin_decode_fuzzer nbin_decode_fuzzer.cc)

llvm_update_compile_flags(nbin_decode_fuzzer)

target_compile_options(nbin_decode_fuzzer PRIVATE -fsanitize=fuzzer-no-link)
target_link_options(nbin_decode_fuzzer PRIVATE -fsanitize=fuzzer)

target_link_libraries(nbin_decode_fuzzer PRIVATE NPUEncoding)

# The seed corpus is committed beside this file. It is copied rather than
# referenced so that a run cannot write new inputs into the source tree: a
# corpus that grows during a run is a corpus somebody commits by accident.
add_custom_target(nbin-fuzz-corpus
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_BINARY_DIR}/fuzz-corpus
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/corpus
                ${CMAKE_BINARY_DIR}/fuzz-corpus
        COMMENT "Copying the seed corpus to ${CMAKE_BINARY_DIR}/fuzz-corpus"
        VERBATIM)

add_dependencies(nbin_decode_fuzzer nbin-fuzz-corpus)
