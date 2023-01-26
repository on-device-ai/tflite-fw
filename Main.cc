#include "Converter.h"

int main(int argc, char *argv[]) {
    
    if (argc < 2) {
        printf("Usage: %s modelFile.tflite [outFile.c]\n",argv[0]);
        return 1;
    }
    
    std::string tflite_model_file_name = argv[1];
    
    std::string output_file_name = "tflite_fw_c_code.c";
    if (argc == 3) {
        output_file_name = argv[2];
    }
    
    if (!tflite_firmware::ConverterFile(tflite_model_file_name, output_file_name)) {
        return 1;
    }

    return 0;
}
