cmake_minimum_required(VERSION 3.14)

if (NOT EMSCRIPTEN)
    return()
endif()

set(TARGET wasm-llama)

add_executable(${TARGET} wasm_llama.cpp)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/common
)

target_link_libraries(${TARGET} PRIVATE llama ggml)

# EXPORTED_FUNCTIONS must match the extern "C" symbols (prefixed with _)
set(WASM_EXPORTS
    "_wasm_llama_init"
    "_wasm_llama_health"
    "_wasm_llama_chat_completion"
    "_wasm_llama_free_str"
    "_malloc"
    "_free"
)
string(JOIN "," WASM_EXPORTS_STR ${WASM_EXPORTS})

set(WASM_RUNTIME_METHODS
    "cwrap"
    "UTF8ToString"
    "stringToUTF8"
    "lengthBytesUTF8"
    "allocate"
    "intArrayFromString"
    "ALLOC_NORMAL"
)
string(JOIN "," WASM_RUNTIME_STR ${WASM_RUNTIME_METHODS})

# NOTE: pthreads require ALL objects compiled with -pthread at cmake-configure time.
# Single-threaded artifact for this PR; MTP threading lands when #739 adds global -pthread.
target_link_options(${TARGET} PRIVATE
    -sEXPORTED_FUNCTIONS=[${WASM_EXPORTS_STR}]
    -sEXPORTED_RUNTIME_METHODS=[${WASM_RUNTIME_STR}]
    -sENVIRONMENT=web
    -sINITIAL_MEMORY=536870912  # 512 MiB — enough for 2B GGUF
    -sMAXIMUM_MEMORY=4294967296 # 4 GiB ceiling
    -sFETCH=1                   # emscripten_fetch for future streaming load
    -sWASM=1
    -sALLOW_MEMORY_GROWTH=1
)
