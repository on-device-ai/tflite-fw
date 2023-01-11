#ifndef TFLITE_FW_KERNELS_H
#define TFLITE_FW_KERNELS_H

#include <iostream>

#include "TFLiteMicroBase.h"
#include "CodeWriter.h"

// Support TFLite Micro OP List
#define OP_FULLY_CONNECTED_FLOAT    1
#define OP_FULLY_CONNECTED_INT8     2
#define TOTAL_OP_SUPPORT            3

namespace tflite_firmware {
    
class FirmwareKernels {
    public:
        FirmwareKernels(CodeWriter *code_writer) : 
            _wr(code_writer)
        {
            for(size_t i = 0; i < TOTAL_OP_SUPPORT; i++) {
                _support_op_list[i] = false;
            }
        }
        void WriteFirmwareKernelImplement(tflite::BuiltinOperator code,TfLiteType tensor_type);
        void WriteFirmwareNode(tflite::BuiltinOperator code,TfLiteType tensor_type,struct NodeInfo *node_info);
        
    private:
        CodeWriter *_wr = nullptr;
        bool _support_op_list[TOTAL_OP_SUPPORT];
        bool _gemmlowp_fixedpoint_wrote = false;
};

}  // namespace tflite_firmware

#endif

