#include "Converter.h"

// For Test Only
int main(int argc, char *argv[]) {
#ifdef NO_QUANT_MODEL
    if (!tflite_firmware::ConverterFile("model_no_quant.tflite", "tflite_fw_c_code.c")) {
        return 1;
    }
#else
    if (!tflite_firmware::ConverterFile("model.tflite", "tflite_fw_c_code.c")) {
        return 1;
    }
#endif
    return 0;
}
