#include "Components/TransientDetector/TransientDetector.hpp"
#include "Fw/Types/Assert.hpp"

namespace Components {

    TransientDetector::TransientDetector(const char* const compName) :
        TransientDetectorComponentBase(compName),
        m_model(nullptr),
        m_interpreter(nullptr)
    {
        // 1. Map the model into a usable data structure. This doesn't copy the data.
        m_model = tflite::GetModel(g_model_data);
        if (m_model->version() != TFLITE_SCHEMA_VERSION) {
            FW_ASSERT(0); // Version mismatch
        }

        // 2. Register required ops (Update to match exact ops used in CNN block)
        m_op_resolver.AddConv2D();
        m_op_resolver.AddDepthwiseConv2D();
        m_op_resolver.AddAveragePool2D(); // For Global Average Pooling
        m_op_resolver.AddReshape();
        m_op_resolver.AddFullyConnected(); // Dense output
        m_op_resolver.AddSoftmax();

        // 3. Build an interpreter to run the model
        static tflite::MicroInterpreter static_interpreter(
            m_model, m_op_resolver, m_tensor_arena, kTensorArenaSize);
            
        m_interpreter = &static_interpreter;

        // 4. Allocate memory from the tensor_arena for the model's tensors
        TfLiteStatus allocate_status = m_interpreter->AllocateTensors();
        if (allocate_status != kTfLiteOk) {
            FW_ASSERT(0); // Failed to allocate Tensors. Increase Arena size.
        }
    }

    TransientDetector::~TransientDetector() {}

    void TransientDetector::init(const NATIVE_INT_TYPE queueDepth, const NATIVE_INT_TYPE instance) {
        TransientDetectorComponentBase::init(queueDepth, instance);
    }

    void TransientDetector::imageIn_handler(const NATIVE_INT_TYPE portNum, Fw::Buffer &fwBuffer) {
        // Assert valid buffer and interpreter
        FW_ASSERT(fwBuffer.getData() != nullptr);
        FW_ASSERT(m_interpreter != nullptr);

        // Get Input Tensor from interpreter
        TfLiteTensor* input = m_interpreter->input(0);
        
        // Ensure buffer matches expected capacity (64x64x1 byte = 4096 bytes)
        // Since input is INT8, size is 4096 bytes.
        FW_ASSERT(fwBuffer.getSize() >= 4096);

        // Copy raw image bytes into the input tensor
        // Note: RP2350 cameras often output unsigned int8 [0, 255]. 
        // If input INT8 tensor expects [-128, 127], subtract 128 from each pixel.
        uint8_t* raw_pixels = reinterpret_cast<uint8_t*>(fwBuffer.getData());
        int8_t* model_input = input->data.int8;
        
        for (int i = 0; i < 4096; i++) {
            model_input[i] = static_cast<int8_t>(raw_pixels[i] - 128); 
        }

        // Run inference
        TfLiteStatus invoke_status = m_interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            // Inference failed
            return;
        }

        // Retrieve output tensor
        TfLiteTensor* output = m_interpreter->output(0);
        
        // Output tensor has shape [1, 2]. Elements: [0]=Normal, [1]=Transient
        int8_t output_quant = output->data.int8[1]; // Get class [1] confidence
        
        // Dequantize the output integer to an actual float probability
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        float confidence = (output_quant - zero_point) * scale;

        // Emit Telemetry
        this->tlmWrite_ConfidenceScore(confidence);

        // Emit Event if confidence > 0.85
        if (confidence > 0.85f) {
            this->log_ACTIVITY_HI_TransientDetected(confidence);
        }

        // Return the buffer logic
        if (this->isConnected_imageOut_OutputPort(0)) {
            this->imageOut_out(0, fwBuffer);
        }
    }
} // end namespace Components
