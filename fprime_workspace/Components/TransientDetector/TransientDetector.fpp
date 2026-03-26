module Components {

    @ Component predicting transient events via CNN TFLite logic
    active component TransientDetector {

        // ----------------------------------------------------------------------
        // Ports
        // ----------------------------------------------------------------------

        @ Input port to receive raw 64x64 grayscale buffer.
        async input port imageIn: Fw.BufferSend

        @ Output port to return the original buffer when done
        output port imageOut: Fw.BufferSend

        // ----------------------------------------------------------------------
        // Telemetry
        // ----------------------------------------------------------------------

        @ Confidence score that a transient event is present 
        telemetry ConfidenceScore: F32

        // ----------------------------------------------------------------------
        // Events
        // ----------------------------------------------------------------------

        @ Event indicating a potential transient was detected
        event TransientDetected(
            confidence: F32
        ) \
        severity activity high \
        format "Transient event detected with confidence {}"

    }
}
