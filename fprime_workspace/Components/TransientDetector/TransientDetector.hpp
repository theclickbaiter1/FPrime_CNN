#ifndef TRANSIENTDETECTOR_HPP
#define TRANSIENTDETECTOR_HPP

#include "Components/TransientDetector/TransientDetectorComponentAc.hpp"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

namespace Components {

    class TransientDetector : public TransientDetectorComponentBase {
        public:
            TransientDetector(const char* const compName);
            ~TransientDetector();

            void init(const NATIVE_INT_TYPE queueDepth, const NATIVE_INT_TYPE instance = 0);

        protected:
            // Handler for incoming 64x64 grayscale image buffers
            void imageIn_handler(
                const NATIVE_INT_TYPE portNum,
                Fw::Buffer &fwBuffer
            ) override;

        private:
            // TensorFlow Lite Micro runtime objects
            const tflite::Model* m_model;
            tflite::MicroInterpreter* m_interpreter;

            // Tensor Arena — memory budget for intermediate activations and weights
            // 120 KB fits within RP2350's 520 KB SRAM, leaving ~400 KB for F Prime services
            constexpr static int kTensorArenaSize = 120 * 1024;
            alignas(16) uint8_t m_tensor_arena[kTensorArenaSize];

            // Op Resolver — 8 ops needed for QAT INT8 model:
            // Conv2D, DepthwiseConv2D, AvgPool, Reshape, FC, Softmax, Quantize, Dequantize
            tflite::MicroMutableOpResolver<8> m_op_resolver;
    };

} // end namespace Components

#endif
