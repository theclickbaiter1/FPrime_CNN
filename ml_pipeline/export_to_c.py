import binascii
import os

def xxd_c_dump(tflite_path, header_path, array_name="g_model_data"):
    """
    Converts a .tflite binary file into a C/C++ header containing a byte array.
    """
    with open(tflite_path, "rb") as f:
        model_data = f.read()

    hex_data = binascii.hexlify(model_data).decode('utf-8')
    hex_array = [f"0x{hex_data[i:i+2]}" for i in range(0, len(hex_data), 2)]
    
    array_len = len(hex_array)
    array_str = ",\n    ".join([", ".join(hex_array[i:i+12]) for i in range(0, array_len, 12)])

    header_content = f"""// Auto-generated from {os.path.basename(tflite_path)}
#ifndef TRANSIENT_MODEL_DATA_H
#define TRANSIENT_MODEL_DATA_H

#ifdef __cplusplus
extern "C" {{
#endif

// Model size in bytes: {array_len}
const unsigned int {array_name}_len = {array_len};
const unsigned char {array_name}[] __attribute__((aligned(4))) = {{
    {array_str}
}};

#ifdef __cplusplus
}}
#endif

#endif // TRANSIENT_MODEL_DATA_H
"""

    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"Exported C-header to {header_path}")

if __name__ == "__main__":
    xxd_c_dump('transient_cnn_int8.tflite', '../fprime_workspace/Components/TransientDetector/model_data.h')
