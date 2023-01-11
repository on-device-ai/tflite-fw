#include "Converter.h"

#include <memory>
#include <fstream>
#include <regex>
#include <vector>

#include "CodeWriter.h"
#include "FirmwareKernels.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/c/builtin_op_data.h"

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#ifndef SUFFICIENT_ARENA_SIZE
#define SUFFICIENT_ARENA_SIZE (128*1024*1024)
#endif

bool tflite_firmware::ConverterFile(const std::string &modelFileName,const std::string &outFileName) {
    // Load model flatbuffer.
    std::ifstream model_file(modelFileName, std::ios::binary | std::ios::ate);
    if (!model_file) {
        std::cerr << "Could not open " << modelFileName << " for read\n";
        return false;
    }
    auto sz = model_file.tellg();
    if (sz == std::ifstream::pos_type(-1)) {
        std::cerr << "Failed to read model file size\n";
        return false;
    }
    std::vector<char> model_data(sz);
    model_file.seekg(0, std::ios::beg);
    if (!model_file.read(model_data.data(), sz)) {
        std::cerr << "Failed to read model file\n";
        return false;
    }

    std::ofstream outFile(outFileName);
    if (!outFile) {
        std::cerr << "Failed to create output file\n";
        return false;
    }

    try {
        Converter converter(model_data.data());
        converter.writeSource(outFile);
        return true;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown exception\n";
    }
    
    return false;
}

tflite_firmware::Converter::Converter(const void *modelData) {
    if (!init(modelData)) {
        throw std::runtime_error("Could not set up converter");
    }
}

bool tflite_firmware::Converter::init(const void *modelData) {
    model_ = tflite::GetModel(modelData);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        errReporter().Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model_->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    auto subgraphs = model_->subgraphs();
    if (subgraphs->size() != 1) {
        std::cerr << "Model needs to have exactly one subgraph as expected by TF "
                    "Lite for Micro\n";
        return false;
    }
    subgraph_ = (*subgraphs)[0];
    auto tensors = subgraph_->tensors();
    if (subgraph_->inputs()->size() == 0 || subgraph_->outputs()->size() == 0) {
        std::cerr << "No inputs or no outputs found in model\n";
        return false;
    }
    for (auto inIndex : *subgraph_->inputs()) {
        inputTensorIndices_.push_back(inIndex);
    }
    for (auto outIndex : *subgraph_->outputs()) {
        outputTensorIndices_.push_back(outIndex);
    }

    // Build an interpreter to run the model with.
    arena_buf_.resize(SUFFICIENT_ARENA_SIZE);
    interpreter_ = std::unique_ptr<tflite::MicroInterpreter>(
        new tflite::MicroInterpreter(
            model_, resolver_, arena_buf_.data(), arena_buf_.size(),
            &microErrReporter_));

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter_->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        errReporter().Report("AllocateTensors() failed");
        return false;
    }

    ptrdiff_t ramTensorBufferSize = 0;
    auto numTensors = tensors->size();
    for (size_t i = 0; i < numTensors; i++) {
        auto tensor = GetTensor(interpreter_.get(), i);
        tensors_.push_back({tensor});
        if (tensor->allocation_type != kTfLiteMmapRo) {   
            ptrdiff_t offset = (uint8_t *)tensor->data.data - arena_buf_.data();
            ptrdiff_t highSize = offset + tensor->bytes;
            ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
        }
    }

    for (size_t i = 0; i < interpreter_->operators_size(); i++) {
        auto nodeAndReg = interpreter_->node_and_registration(i);
        auto node       = &nodeAndReg.node;
        auto reg        = nodeAndReg.registration;
        auto code       = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

        printf("operation %lu: %s\n", i, tflite::EnumNamesBuiltinOperator()[code]);

        RegistrationInfo regInfo;
        regInfo.reg = reg;
        regInfo.code = code;
        auto itOp = std::find(registrations_.begin(), registrations_.end(), regInfo);
        if (itOp == registrations_.end()) {
            itOp = registrations_.insert(registrations_.end(), regInfo);
        }

        // There doesn't seem to be a way to get the node pointer, so copy it.
        nodes_.push_back(NodeInfo{*node, itOp - registrations_.begin()});
    }

    arenaBufferSize_ = ramTensorBufferSize;
    arenaBufferSize_ = (arenaBufferSize_ % 16) ? ((arenaBufferSize_ / 16) + 1) * 16 : arenaBufferSize_;

    return true;
}

void tflite_firmware::Converter::writeSource(std::ostream &out) {
    
    CodeWriter wr(out);
    FirmwareKernels kernels(&wr);
    
    wr << R"(
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
)";
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto &node = nodes_[i].node;
        auto &regInfo = registrations_[nodes_[i].regIndex];
        
        kernels.WriteFirmwareKernelImplement(regInfo.code,tensors_[node.inputs->data[0]].tensor->type);
        
    }
    
wr << R"(
#define TENSOR_ARENA_SIZE )"
   << arenaBufferSize_ << R"(
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

)";
  
    for (size_t i = 0; i < tensors_.size(); i++) {
        auto &t = tensors_[i].tensor;
        if (t->allocation_type == kTfLiteMmapRo) {
            wr.writeTensor(*t, "tensor" + std::to_string(i) + "_data" );
        } else {
            switch (t->type) {
                case kTfLiteFloat32:
                    wr << "float* tensor" << std::to_string(i) << "_data = (float *)" << "tensor_arena + " << ((uintptr_t)t->data.data - (uintptr_t)arena_buf_.data()) << "; /* tensor size = " << t->bytes << " bytes*/ \n";
                    break;
                case kTfLiteInt8:
                    wr << "int8_t* tensor" << std::to_string(i) << "_data = (int8_t *)" << "tensor_arena + " << ((uintptr_t)t->data.data - (uintptr_t)arena_buf_.data()) << "; /* tensor size = " << t->bytes << " bytes*/ \n";
                    break;
            }
        }
        wr.writeIntArray(*t->dims, "tensor" + std::to_string(i) + "_ndim" , "tensor" + std::to_string(i) + "_dims" );
        wr.writeQuantization(t->quantization, "quant" + std::to_string(i));
        wr << "\n";
    }
    wr << "void  model_invoke() {" << "\n";
    wr << "\n    /* input tensor index = " << inputTensorIndices_.front() << " , output tensor index = " << outputTensorIndices_.front() << " */ \n\n";
    
    for (size_t i = 0; i < nodes_.size(); i++) {
        auto &node = nodes_[i].node;
        auto &regInfo = registrations_[nodes_[i].regIndex];
        
        kernels.WriteFirmwareNode(regInfo.code,tensors_[node.inputs->data[0]].tensor->type,&nodes_[i]);
        
    }
    
    wr << "}" << "\n";
}

