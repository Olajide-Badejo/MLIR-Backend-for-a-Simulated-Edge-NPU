# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The generation wiring of Section 9.4. One description file goes in and six
# artifacts come out, and every layer of the encoder reads one of them rather
# than carrying its own copy of the instruction set.
#
# These are plain custom commands rather than the `tablegen()` macro from
# TableGen.cmake. That macro is built around `${project}_TABLEGEN_EXE` and the
# conventions of an in tree LLVM subproject, and using it here would mean
# setting three cache variables to make a four line command look like it
# belonged to LLVM. The dependency edges are the same either way and they are
# visible here.

set(NPU_ISA_DESCRIPTION ${CMAKE_CURRENT_SOURCE_DIR}/NPUISADescription.td)

# Each artifact, with the action that writes it and a one line reason it
# exists. The reasons are here because a generated file with no stated consumer
# is a file somebody deletes.
#
#   NPUISAOpcodes.inc     the Opcode enum and kMaxOpcode, so appending an
#                         opcode cannot be done halfway
#   NPUISAOpcodeInfo.inc  the arity, field presence, memory space and element
#                         type rules Program::validate() reads
#   NPUISAChecks.inc      the check names of Section 9.2, so the manual takes
#                         them from the source rather than copying them
#   NPUISADispatch.def    the simulator's dispatch skeleton, which P7 expands
#   NPUISAManual.md       the generated sections of docs/ISA_MANUAL.md
#   NPUISAOpcodes.json    the opcode list scripts/check-reachability.py reads
set(NPU_ISA_ARTIFACTS
        "NPUISAOpcodes.inc:-gen-opcode-enum"
        "NPUISAOpcodeInfo.inc:-gen-opcode-info"
        "NPUISAChecks.inc:-gen-check-enum"
        "NPUISADispatch.def:-gen-dispatch-def"
        "NPUISAManual.md:-gen-manual"
        "NPUISAOpcodes.json:-gen-json"
        )

set(NPU_ISA_GENERATED_FILES "")
foreach(entry IN LISTS NPU_ISA_ARTIFACTS)
  string(REPLACE ":" ";" parts "${entry}")
  list(GET parts 0 output)
  list(GET parts 1 action)
  add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${output}
          COMMAND $<TARGET_FILE:npu-isa-tblgen>
                  ${action}
                  -I ${NPU_SOURCE_DIR}/include
                  ${NPU_ISA_DESCRIPTION}
                  -o ${CMAKE_CURRENT_BINARY_DIR}/${output}
          DEPENDS npu-isa-tblgen ${NPU_ISA_DESCRIPTION}
          COMMENT "Generating ${output} from the ISA description"
          VERBATIM)
  list(APPEND NPU_ISA_GENERATED_FILES ${CMAKE_CURRENT_BINARY_DIR}/${output})
endforeach()

# The target every consumer depends on. Naming it *IncGen matches what
# add_mlir_dialect produces for the dialects, so the two generation mechanisms
# in this project look the same from a CMakeLists that consumes them.
add_custom_target(NPUEncodingIncGen DEPENDS ${NPU_ISA_GENERATED_FILES})

# Exported so the top level npu-isa-doc target and the staleness gate can name
# the two files they copy without rebuilding the paths.
set(NPU_ISA_GENERATED_MANUAL ${CMAKE_CURRENT_BINARY_DIR}/NPUISAManual.md
        CACHE INTERNAL "The generated sections of docs/ISA_MANUAL.md")
set(NPU_ISA_GENERATED_JSON ${CMAKE_CURRENT_BINARY_DIR}/NPUISAOpcodes.json
        CACHE INTERNAL "The generated opcode list")
