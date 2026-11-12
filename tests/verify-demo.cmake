file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(REMOVE "${OUTPUT_DIR}/scores.txt")
execute_process(
    COMMAND "${VIDEOPROC_EXE}" "${DEMO_DIR}/signal-reference.avi"
        "${DEMO_DIR}/signal-decale.avi" 17100 20103
    WORKING_DIRECTORY "${OUTPUT_DIR}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 30
)
if(NOT status EQUAL 0 OR NOT EXISTS "${OUTPUT_DIR}/scores.txt")
    message(FATAL_ERROR "Échec de la démo (${status}).\n${output}\n${errors}")
endif()
file(READ "${OUTPUT_DIR}/scores.txt" actual)
file(READ "${DEMO_DIR}/scores-ajustement.txt" expected)
string(STRIP "${actual}" actual)
string(STRIP "${expected}" expected)
if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "L'alignement exporté diffère de la référence.\n${actual}")
endif()
