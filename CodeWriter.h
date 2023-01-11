#ifndef TFLITE_FW_CODEWRITER_H
#define TFLITE_FW_CODEWRITER_H

#include <iostream>

#include "tensorflow/lite/micro/micro_interpreter.h"

namespace tflite_firmware {

// Helper functions for top-level code generation.
class CodeWriter {
    public:
        CodeWriter(std::ostream &out);

        // Write IntArray with variable declaration.
        void writeIntArray(const TfLiteIntArray &arr, const std::string &ndim_name,const std::string &dims_name);
    
        void writeTensor(const TfLiteTensor &t, const std::string &name);
        
        void writeQuantization(const TfLiteQuantization &q, const std::string &name);

        template <typename T>
        CodeWriter &operator<<(T &&value) {
            out_ << std::forward<T>(value);
            return *this;
        }
        
    private:
        // Write only the comma separated contents of an IntArray.
        void writeIntArrayData(const TfLiteIntArray &arr);

    private:
        std::ostream &out_;
};

}  // namespace tflite_firmware

#endif
