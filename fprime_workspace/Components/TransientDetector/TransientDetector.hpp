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
            // ----------------------------------------------------------------------
            // Handler implementations for user-defined typed input ports
            // ----------------------------------------------------------------------
            void imageIn_handler(
                const NATIVE_INT_TYPE portNum,
                Fw::Buffer &fwBuffer
            ) override;

        private:
            // ----------------------------------------------------------------------
            // TensorFlow Lite Micro variables
            // ----------------------------------------------------------------------
            const tflite::Model* m_model;
            tflite::MicroInterpreter* m_interpreter;

            // Tensor Arena (Memory budget to fit intermediate features & weights)
            // Assigned 120KB to be safely within the 150KB limit bounds for the RP2350
            constexpr static int kTensorArenaSize = 120 * 1024;
            alignas(16) uint8_t m_tensor_arena[kTensorArenaSize];

            // TFLite Resolver holds the subset of operations the MCU supports
            // Count must cover: Conv2D, DepthwiseConv2D, AveragePool2D, Reshape, FullyConnected, Softmax
            tflite::MicroMutableOpResolver<6> m_op_resolver;
    };

} // end namespace Components

#endif
