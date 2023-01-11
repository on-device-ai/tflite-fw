#include "FirmwareKernels.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"

void tflite_firmware::FirmwareKernels::WriteFirmwareKernelImplement(tflite::BuiltinOperator code,TfLiteType tensor_type) {

    if(code == tflite::BuiltinOperator_FULLY_CONNECTED) {
        if(tensor_type == kTfLiteFloat32) {
            if(_support_op_list[OP_FULLY_CONNECTED_FLOAT] == false) {                 

*_wr << R"(
/* FULLY_CONNECTED OP : Floating point implementation */
void FULLY_CONNECTED_float(
    const float* input, int32_t input_dim, const int32_t* input_shape,
    const float* filter, int32_t filter_dim, const int32_t* filter_shape,
    const float* bias, int32_t bias_dim, const int32_t* bias_shape,
    float* output, int32_t output_dim, const int32_t* output_shape,
    float output_activation_min,float output_activation_max)
{
    int32_t flat_size = 1;
    int32_t batches = 0;
    int32_t output_depth = 0;
    int32_t accum_depth = 0;
    int     i,b,out_c,d;
    
    for (i = 0; i < output_dim; ++i) {
        flat_size *= (i == (output_dim-1)) ? 1 : output_shape[i];
    }
    batches = flat_size;
    if(filter_shape[filter_dim-2] <= output_shape[output_dim-1]) {
        output_depth = filter_shape[filter_dim-2];
    } else {
        output_depth = output_shape[output_dim-1];
    }
    accum_depth = filter_shape[filter_dim-1];

    for (b = 0; b < batches; ++b) {
        for (out_c = 0; out_c < output_depth; ++out_c) {
            float total = 0.0f;
            for (d = 0; d < accum_depth; ++d) {
                total += input[b * accum_depth + d] * filter[out_c * accum_depth + d];
            }
            total += bias[out_c];
            total = (total > output_activation_min) ? total : output_activation_min;
            total = (total < output_activation_max) ? total : output_activation_max;
            output[out_c + output_depth * b] = total;
            
        }
    }  
}
)";

                _support_op_list[OP_FULLY_CONNECTED_FLOAT] = true;
            }
                
        } else if(tensor_type == kTfLiteInt8) {
            if(_support_op_list[OP_FULLY_CONNECTED_INT8] == false) {
                if(_gemmlowp_fixedpoint_wrote == false) {

*_wr << R"(
/* --- gemmlowp : fixedpoint --- */

#include <assert.h>

static int32_t BitAnd(int32_t a, int32_t b) {
    return a & b;
}

static int32_t BitNot(int32_t a) {
    return ~a;
}

static int32_t Add(int32_t a, int32_t b) {
    return a + b;
}

static int32_t ShiftRight(int32_t a, int offset) {
    return a >> offset;
}

static int32_t MaskIfNonZero(int32_t a) {
    return a ? BitNot(0) : 0;
}

static int32_t MaskIfGreaterThan(int32_t a, int32_t b) {
    return MaskIfNonZero(a > b);
}

static int32_t MaskIfLessThan(int32_t a, int32_t b) {
    return MaskIfNonZero(a < b);
}

static int32_t SaturatingRoundingDoublingHighMul(int32_t a,int32_t b) {
    bool overflow = ((a == b) && (a == INT32_MIN));
    int64_t a_64 = (int64_t)a;
    int64_t b_64 = (int64_t)b;
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (1ll << 31));
    return overflow ? INT32_MAX : ab_x2_high32;
}

static int32_t RoundingDivideByPOT(int32_t x, int exponent) {
    assert(exponent >= 0);
    assert(exponent <= 31);
    const int32_t mask = ((1ll << exponent) - 1);
    const int32_t zero = 0;
    const int32_t one =  1;
    const int32_t remainder = BitAnd(x, mask);
    const int32_t threshold = Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
    return Add(ShiftRight(x, exponent), BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}

static int32_t MultiplyByQuantizedMultiplier(int32_t x,int32_t quantized_multiplier,int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    
    return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),right_shift);
}

/* ----------------------------- */
)";
                                             
                    _gemmlowp_fixedpoint_wrote = true;
                }
                
*_wr << R"(
/* FULLY_CONNECTED OP : Quantized 8-bit integer implementation */
void FULLY_CONNECTED_int8(
    const int8_t* input, int32_t input_dim, const int32_t* input_shape,
    const int8_t* filter, int32_t filter_dim, const int32_t* filter_shape,
    const int32_t* bias, int32_t bias_dim, const int32_t* bias_shape,
    int8_t* output, int32_t output_dim, const int32_t* output_shape,
    int32_t input_offset, int32_t filter_offset,
    int32_t output_offset, int32_t output_multiplier, int output_shift,
    int32_t output_activation_min, int32_t output_activation_max)
{
    int32_t flat_size = 1;
    int32_t batches = 0;
    int32_t output_depth = 0;
    int32_t accum_depth = 0;
    int     i,b,out_c,d;
    
    for (i = 0; i < output_dim; ++i) {
        flat_size *= (i == (output_dim-1)) ? 1 : output_shape[i];
    }
    batches = flat_size;
    if(filter_shape[filter_dim-2] <= output_shape[output_dim-1]) {
        output_depth = filter_shape[filter_dim-2];
    } else {
        output_depth = output_shape[output_dim-1];
    }
    accum_depth = filter_shape[filter_dim-1];
    
    for (b = 0; b < batches; ++b) {
        for (out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (d = 0; d < accum_depth; ++d) {
                int32_t input_val = input[b * accum_depth + d];
                int32_t filter_val = filter[out_c * accum_depth + d];
                acc += (filter_val + filter_offset) * (input_val + input_offset);
            }
            acc += bias[out_c];
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
            acc += output_offset;
            acc = (acc > output_activation_min) ? acc : output_activation_min;
            acc = (acc < output_activation_max) ? acc : output_activation_max;
            output[out_c + output_depth * b] = (int8_t)acc;
        }
    } 
}                
)";                                 
               
                _support_op_list[OP_FULLY_CONNECTED_INT8] = true;
            }
        }
    }    
}

void tflite_firmware::FirmwareKernels::WriteFirmwareNode(tflite::BuiltinOperator code,TfLiteType tensor_type,struct NodeInfo *node_info) {
     auto &node = node_info->node;
     
     if(code == tflite::BuiltinOperator_FULLY_CONNECTED) {
        uint8_t const *opdata = (uint8_t const *) node.builtin_data;
           
        if(tensor_type == kTfLiteFloat32) {
            
            *_wr << "    FULLY_CONNECTED_float(" << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[0]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[0]) << "_ndim," << "tensor" << std::to_string(node.inputs->data[0]) << "_dims," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[1]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[1]) << "_ndim," << "tensor" << std::to_string(node.inputs->data[1]) << "_dims," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[2]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[2]) << "_ndim," << "tensor" << std::to_string(node.inputs->data[2]) << "_dims," << "\n";
            *_wr << "        tensor" << std::to_string(node.outputs->data[0]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.outputs->data[0]) << "_ndim," << "tensor" << std::to_string(node.outputs->data[0]) << "_dims," << "\n";
            if(opdata[0] == kTfLiteActNone) {
                *_wr << "        " << std::numeric_limits<float>::lowest() << "," << std::numeric_limits<float>::max() << ");" << "\n" << "\n";
            } else if (opdata[0] == kTfLiteActRelu) {
                *_wr << "        " << 0.0f << "," << std::numeric_limits<float>::max() << ");" << "\n" << "\n";
            }
                
        } else if(tensor_type == kTfLiteInt8) {
                
            auto* data = static_cast<tflite::OpDataFullyConnected*>(node.user_data);

            *_wr << "    FULLY_CONNECTED_int8(" << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[0]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[0]) << "_ndim," << "tensor" << std::to_string(node.inputs->data[0]) << "_dims," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[1]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[1]) << "_ndim," << "tensor" << std::to_string(node.inputs->data[1]) << "_dims," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[2]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.inputs->data[2]) << "_ndim," << "tensor" << std::to_string(node.inputs->data[2]) << "_dims," << "\n";
            *_wr << "        tensor" << std::to_string(node.outputs->data[0]) << "_data," << "\n";
            *_wr << "        tensor" << std::to_string(node.outputs->data[0]) << "_ndim," << "tensor" << std::to_string(node.outputs->data[0]) << "_dims," << "\n";
            *_wr << "        " << -data->input_zero_point << "," << -data->filter_zero_point << ",\n";
            *_wr << "        " << data->output_zero_point << "," << data->output_multiplier << "," << data->output_shift << ",\n";
            *_wr << "        " << data->output_activation_min << "," << data->output_activation_max << ");" << "\n" << "\n";
        }
    }
}

