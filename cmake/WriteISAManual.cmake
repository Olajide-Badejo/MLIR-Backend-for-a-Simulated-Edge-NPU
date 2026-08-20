# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The copy step of the npu-isa-doc target. Run in script mode with
# -DGENERATED=, -DOUTPUT=, -DJSON_GENERATED= and -DJSON_OUTPUT=.
#
# It differs from WriteDialectReference.cmake in exactly one way, and the
# difference is the reason it is a second script rather than a parameter on the
# first. The dialect reference is generated end to end, so writing it is a
# copy. The ISA manual is mostly hand written prose about a byte order policy,
# a version policy and a memory model, with generated tables inside it, so
# writing it is a splice: the text between the BEGIN and END markers is
# replaced and everything else is left exactly as its author wrote it.
#
# Determinism matters here for the same reason it does there. This file's whole
# purpose is to be diffed by CI, so anything that varies between two runs on
# the same input turns the staleness gate into a coin toss. Nothing below
# depends on the clock, the path, or the filesystem's iteration order.

if(NOT EXISTS "${GENERATED}")
  message(FATAL_ERROR
    "The generated manual sections were not found at ${GENERATED}. The "
    "npu-isa-doc target depends on NPUEncodingIncGen, so this means the "
    "generation itself did not run.")
endif()

if(NOT EXISTS "${OUTPUT}")
  message(FATAL_ERROR
    "docs/ISA_MANUAL.md was not found at ${OUTPUT}. The manual is hand written "
    "prose with generated tables spliced into it, so this script replaces "
    "sections of an existing file and cannot create one from nothing.")
endif()

file(READ "${GENERATED}" generated)
file(READ "${OUTPUT}" manual)

# The generated file holds one or more marked sections. Each is spliced into
# the manual at the matching pair of markers, and a marker the manual does not
# carry is a hard error rather than a silently skipped section: a manual that
# lost a marker is a manual whose table stopped being regenerated, which is
# exactly the drift this whole mechanism exists to prevent.
set(sections "opcode table" "validation checks")

foreach(section IN LISTS sections)
  set(begin_marker "<!-- BEGIN GENERATED: ${section} -->")
  set(end_marker "<!-- END GENERATED: ${section} -->")

  string(FIND "${generated}" "${begin_marker}" gen_begin)
  string(FIND "${generated}" "${end_marker}" gen_end)
  if(gen_begin LESS 0 OR gen_end LESS 0)
    message(FATAL_ERROR
      "The generated file does not carry the '${section}' section. Either the "
      "generator's -gen-manual action stopped emitting it or this script's "
      "section list is stale.")
  endif()
  string(LENGTH "${end_marker}" end_len)
  math(EXPR gen_len "${gen_end} + ${end_len} - ${gen_begin}")
  string(SUBSTRING "${generated}" ${gen_begin} ${gen_len} block)

  string(FIND "${manual}" "${begin_marker}" man_begin)
  string(FIND "${manual}" "${end_marker}" man_end)
  if(man_begin LESS 0 OR man_end LESS 0)
    message(FATAL_ERROR
      "docs/ISA_MANUAL.md does not carry the markers for the '${section}' "
      "section. The manual must contain both\n"
      "    ${begin_marker}\n"
      "    ${end_marker}\n"
      "for this script to know where the generated table goes.")
  endif()
  math(EXPR man_len "${man_end} + ${end_len} - ${man_begin}")
  string(SUBSTRING "${manual}" 0 ${man_begin} head)
  math(EXPR tail_start "${man_begin} + ${man_len}")
  string(SUBSTRING "${manual}" ${tail_start} -1 tail)
  set(manual "${head}${block}${tail}")
endforeach()

file(WRITE "${OUTPUT}" "${manual}")
message(STATUS "Wrote ${OUTPUT}")

# The opcode list. Unlike the manual this one is generated end to end, so it is
# a copy with a header of its own. JSON carries no comment syntax, so the
# "do not edit" line is a string field the generator emits rather than a
# comment this script prepends.
if(NOT EXISTS "${JSON_GENERATED}")
  message(FATAL_ERROR
    "The generated opcode list was not found at ${JSON_GENERATED}.")
endif()

file(READ "${JSON_GENERATED}" opcode_json)
file(WRITE "${JSON_OUTPUT}" "${opcode_json}")
message(STATUS "Wrote ${JSON_OUTPUT}")
