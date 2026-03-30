module Components {

    @ CNN-based transient space event detector using TFLite Micro on RP2350
    active component TransientDetector {

        // ----------------------------------------------------------------------
        // Ports
        // ----------------------------------------------------------------------

        @ Input port to receive raw 64x64 grayscale image buffer from camera
        async input port imageIn: Fw.BufferSend

        @ Output port to return the original buffer when inference is complete
        output port imageOut: Fw.BufferSend

        // ----------------------------------------------------------------------
        // Telemetry
        // ----------------------------------------------------------------------

        @ Confidence score that a transient event is present (class 0 probability)
        telemetry ConfidenceScore: F32

        @ Detected class index (0=transient, 1=starfield, 2=bright_source, 3=earth_limb)
        telemetry DetectedClass: U8

        // ----------------------------------------------------------------------
        // Events
        // ----------------------------------------------------------------------

        @ A transient event was detected with high confidence
        event TransientDetected(
            confidence: F32
        ) \
        severity activity high \
        format "TRANSIENT EVENT DETECTED — confidence: {}"

    }
}
