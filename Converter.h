#ifndef TFLITE_FW_CONVERTER_H
#define TFLITE_FW_CONVERTER_H

#include <iostream>

#include "TFLiteMicroUtil.h"

namespace tflite_firmware {

bool ConverterFile(const std::string &modelFileName,const std::string &outFileName);

class Converter {
    public:
        // modelData: Flatbuffer binary data.
        // prefix: This string is prepended to every global name.
        Converter(const void *modelData);
        
        void writeSource(std::ostream &out);
        
    private:
        bool init(const void *modelData);
        tflite::ErrorReporter &errReporter() { return microErrReporter_; }
        
        tflite::MicroErrorReporter microErrReporter_;
        const tflite::Model *model_ = nullptr;
        const tflite::SubGraph *subgraph_ = nullptr;
        tflite::AllOpsResolver resolver_;
        std::vector<uint8_t> arena_buf_;
        std::unique_ptr<tflite::MicroInterpreter> interpreter_;
        
        size_t arenaBufferSize_ = 0;
        std::vector<TensorInfo> tensors_;
        std::vector<RegistrationInfo> registrations_;
        std::vector<NodeInfo> nodes_;
        std::vector<int32_t> inputTensorIndices_;
        std::vector<int32_t> outputTensorIndices_;
};

}  // namespace tflite_firmware

#endif
