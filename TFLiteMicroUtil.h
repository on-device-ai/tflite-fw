#ifndef TFLITE_FW_TFLMUTIL_H
#define TFLITE_FW_TFLMUTIL_H

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite_firmware {
    
struct TensorInfo {
    TensorInfo(const TfLiteTensor *tensor_ptr) :
        tensor(tensor_ptr)
    {}
    const TfLiteTensor *tensor = nullptr;
};

struct RegistrationInfo {
    const TfLiteRegistration *reg = nullptr;
    tflite::BuiltinOperator code;
    bool operator==(const RegistrationInfo &other) {
        if (code != other.code) {
            return false; 
        } 
        return true;
    }
};

struct NodeInfo {
    NodeInfo() {}
    NodeInfo(TfLiteNode tfl_node, ptrdiff_t reg_index) :
        node(tfl_node),
        regIndex(reg_index)
    {}
    TfLiteNode node;
    ptrdiff_t regIndex = -1;
};

TfLiteTensor *GetTensor(tflite::MicroInterpreter *interpreter, int i);

}

#endif
