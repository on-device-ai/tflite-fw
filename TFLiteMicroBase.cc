#include <sstream>
#define private public
#include "tensorflow/lite/micro/micro_interpreter.h"
#undef private

#include "TFLiteMicroBase.h"

TfLiteTensor *tflite_firmware::GetTensor(tflite::MicroInterpreter *interpreter, int i) {
    auto ctx = &interpreter->context_;
    return ctx->GetTensor(ctx, i);
}
