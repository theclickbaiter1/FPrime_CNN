set(SOURCE_FILES
    "${CMAKE_CURRENT_LIST_DIR}/TransientDetector.cpp"
)

# To deploy on RP2350, ensure dependencies align with Pi Pico SDK and F' RTOS.
set(FPP_FILES
    "${CMAKE_CURRENT_LIST_DIR}/TransientDetector.fpp"
)

# NOTE: F' will need to link `tensorflow_lite_micro` as an external library.
# Usually this is added via find_package(tflite_micro) in the parent CMakeLists.txt

set(COMPONENT_DEPENDENCIES
    Fw/Buffer
    Fw/Types
    Fw/Port
)

# Register F Prime module
register_fprime_module()
