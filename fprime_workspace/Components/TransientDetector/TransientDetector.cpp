#include "Components/TransientDetector/TransientDetector.hpp"
#include "Fw/Types/Assert.hpp"

namespace Components {

    // human-readable names matching the CNN output ordering
    static const char* CLASS_LABELS[] = {
        "transient",    // class 0 — triggers event
        "starfield",    // class 1 — ignored
        "bright_source",// class 2 — ignored (Sun / Moon / planet)
        "earth_limb"    // class 3 — ignored (Earth horizon)
    };

    TransientDetector::TransientDetector(const char* const compName) :
        TransientDetectorComponentBase(compName),
        m_model(nullptr),
        m_interpreter(nullptr)
    {
        // 1. Map the model flatbuffer into a usable data structure (zero-copy)
        m_model = tflite::GetModel(g_model_data);
        if (m_model->version() != TFLITE_SCHEMA_VERSION) {
            FW_ASSERT(0); // Version mismatch — model was built with incompatible schema
        }

        // 2. Register the exact ops used by the quantized CNN
        //    Must match ops in the .tflite file exactly.
        //    Missing an op causes AllocateTensors() to silently fail or assert.
        m_op_resolver.AddConv2D();           // Block 1: standard 3x3 convolution
        m_op_resolver.AddDepthwiseConv2D();  // Blocks 2+3: DW separable convolution
        m_op_resolver.AddAveragePool2D();    // GlobalAveragePooling2D impl
        m_op_resolver.AddReshape();          // reshape before Dense
        m_op_resolver.AddFullyConnected();   // Dense(4) output layer
        m_op_resolver.AddSoftmax();          // final probability normalisation
        m_op_resolver.AddQuantize();         // INT8 quantize nodes (PTQ conversion)
        m_op_resolver.AddDequantize();       // INT8 dequantize nodes (PTQ conversion)

        // 3. Build an interpreter to run the model with the allocated tensor arena
        static tflite::MicroInterpreter static_interpreter(
            m_model, m_op_resolver, m_tensor_arena, kTensorArenaSize);
            
        m_interpreter = &static_interpreter;

        // 4. Allocate memory from the tensor_arena for the model's tensors
        TfLiteStatus allocate_status = m_interpreter->AllocateTensors();
        if (allocate_status != kTfLiteOk) {
            FW_ASSERT(0); // Failed — increase kTensorArenaSize in .hpp
        }
    }

    TransientDetector::~TransientDetector() {}

    void TransientDetector::init(const NATIVE_INT_TYPE queueDepth, const NATIVE_INT_TYPE instance) {
        TransientDetectorComponentBase::init(queueDepth, instance);
    }

    void TransientDetector::imageIn_handler(const NATIVE_INT_TYPE portNum, Fw::Buffer &fwBuffer) {
        // Validate inputs
        FW_ASSERT(fwBuffer.getData() != nullptr);
        FW_ASSERT(m_interpreter != nullptr);

        // Get input tensor from interpreter
        TfLiteTensor* input = m_interpreter->input(0);
        
        // Input buffer must be at least 64*64*1 = 4096 bytes
        FW_ASSERT(fwBuffer.getSize() >= 4096);

        // Copy raw camera bytes into the input tensor.
        // Camera outputs uint8 in [0, 255]; INT8 tensor expects [-128, 127].
        // Subtract 128 (i.e. cast to signed and bias-shift): uint8 0 → int8 -128.
        uint8_t* raw_pixels = reinterpret_cast<uint8_t*>(fwBuffer.getData());
        int8_t* model_input = input->data.int8;
        
        for (int i = 0; i < 4096; i++) {
            model_input[i] = static_cast<int8_t>(raw_pixels[i] - 128); 
        }

        // Run inference
        TfLiteStatus invoke_status = m_interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            return; // Inference failed — skip this frame
        }

        // Retrieve 4-class output tensor: [transient, starfield, bright_source, earth_limb]
        // Dequantize each logit from INT8 → float32: value = (raw - zero_point) * scale
        TfLiteTensor* output = m_interpreter->output(0);
        
        // dequantisation parameters set during PTQ calibration
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        
        // argmax over 4 classes
        int best_class = 0;
        float best_confidence = -1.0f;
        float confidences[4];
        
        for (int c = 0; c < 4; c++) {
            confidences[c] = (output->data.int8[c] - zero_point) * scale;
            if (confidences[c] > best_confidence) {
                best_confidence = confidences[c];
                best_class = c;
            }
        }

        // Emit telemetry — always report the transient confidence (class 0)
        // so that ground operators can monitor the raw score over time.
        float transient_confidence = confidences[0];
        this->tlmWrite_ConfidenceScore(transient_confidence);
        this->tlmWrite_DetectedClass(static_cast<U8>(best_class));

        // Fire high-priority event ONLY when class is transient (0)
        // and the confidence exceeds the 0.70 detection threshold.
        // Threshold is conservative: prefer missing faint events over
        // false alarms that waste downlink bandwidth.
        if (best_class == 0 && transient_confidence > 0.70f) {
            this->log_ACTIVITY_HI_TransientDetected(transient_confidence);
        }

        // Return the buffer to sender
        if (this->isConnected_imageOut_OutputPort(0)) {
            this->imageOut_out(0, fwBuffer);
        }
    }
} // end namespace Components
