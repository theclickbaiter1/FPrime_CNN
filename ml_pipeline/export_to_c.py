"""
export_to_c.py — Convert INT8 TFLite model to C header for RP2350 firmware.

Generates model_data.h containing the model as a byte array for
inclusion in the F Prime TransientDetector component.
"""

import binascii
import os
import json


def xxd_c_dump(tflite_path, header_path, array_name="g_model_data"):
    """Convert a .tflite binary file into a C/C++ header byte array."""
    with open(tflite_path, "rb") as f:
        model_data = f.read()

    model_size = len(model_data)
    print(f"Model: {tflite_path}")
    print(f"  Size: {model_size:,} bytes ({model_size/1024:.1f} KB)")

    # Size validation for RP2350
    if model_size > 100 * 1024:
        print(f"  [WARN] Model exceeds 100 KB target for RP2350!")
    else:
        print(f"  [PASS] Model fits within RP2350 flash budget")

    hex_data = binascii.hexlify(model_data).decode('utf-8')
    hex_array = [f"0x{hex_data[i:i+2]}" for i in range(0, len(hex_data), 2)]

    array_len = len(hex_array)
    lines = []
    for i in range(0, array_len, 12):
        lines.append("    " + ", ".join(hex_array[i:i+12]))
    array_str = ",\n".join(lines)

    header_content = f"""// Auto-generated from {os.path.basename(tflite_path)}
// Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)
// Input:  64x64x1 INT8 (4096 bytes)
// Output: 4 classes [transient, starfield, bright_source, earth_limb]
//
// DO NOT EDIT — regenerate with: python3 export_to_c.py

#ifndef TRANSIENT_MODEL_DATA_H
#define TRANSIENT_MODEL_DATA_H

#ifdef __cplusplus
extern "C" {{
#endif

// Number of bytes in the model
const unsigned int {array_name}_len = {array_len};

// Model data aligned for efficient memory access
const unsigned char {array_name}[] __attribute__((aligned(4))) = {{
{array_str}
}};

#ifdef __cplusplus
}}
#endif

#endif // TRANSIENT_MODEL_DATA_H
"""

    os.makedirs(os.path.dirname(header_path) if os.path.dirname(header_path) else '.', exist_ok=True)
    with open(header_path, "w") as f:
        f.write(header_content)

    print(f"  Exported C-header: {header_path}")
    print(f"  Array length: {array_len} bytes")

    return model_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Export TFLite model to C header')
    parser.add_argument('--input', default='output/transient_cnn_int8.tflite')
    parser.add_argument('--output',
                        default='../fprime_workspace/Components/TransientDetector/model_data.h')
    args = parser.parse_args()

    xxd_c_dump(args.input, args.output)
