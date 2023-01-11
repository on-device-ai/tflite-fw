TARGET_TOOLCHAIN_ROOT := 
TARGET_TOOLCHAIN_PREFIX := 

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
CXX             := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)g++'
CC              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)gcc'
AS              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)as'
AR              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ar' 
LD              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)ld'
NM              := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)nm'
OBJDUMP         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objdump'
OBJCOPY         := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)objcopy'
SIZE            := '$(TARGET_TOOLCHAIN_ROOT)$(TARGET_TOOLCHAIN_PREFIX)size'

RM = rm -f
ARFLAGS := -csr

SRCS := \
./edge-impulse-sdk/tensorflow/lite/micro/simple_memory_allocator.cc ./edge-impulse-sdk/tensorflow/lite/micro/memory_helpers.cc ./edge-impulse-sdk/tensorflow/lite/micro/test_helpers.cc ./edge-impulse-sdk/tensorflow/lite/micro/recording_micro_allocator.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_error_reporter.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_time.cc ./edge-impulse-sdk/tensorflow/lite/micro/recording_simple_memory_allocator.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_string.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_profiler.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_utils.cc ./edge-impulse-sdk/porting/posix/ei_classifier_porting.cpp ./edge-impulse-sdk/porting/posix/debug_log.cc ./edge-impulse-sdk/tensorflow/lite/micro/all_ops_resolver.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_interpreter.cc ./edge-impulse-sdk/tensorflow/lite/micro/micro_allocator.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/logistic.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/circular_buffer.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/conv_common.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/conv.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/prelu.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/softmax_common.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/softmax.cc  ./edge-impulse-sdk/tensorflow/lite/micro/kernels/dequantize.cc  ./edge-impulse-sdk/tensorflow/lite/micro/kernels/pad.cc  ./edge-impulse-sdk/tensorflow/lite/micro/kernels/ethosu.cc  ./edge-impulse-sdk/tensorflow/lite/micro/kernels/l2norm.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/tanh.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/activations.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/ceil.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/arg_min_max.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/fully_connected_common.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/fully_connected.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/add.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/add_n.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/split.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/round.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/pack.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/floor.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/unpack.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/svdf_common.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/svdf.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/sub.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/pooling.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/concatenation.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/neg.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/quantize_common.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/quantize.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/mul.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/maximum_minimum.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/reshape.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/reduce.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/strided_slice.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/logical.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/elementwise.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/comparisons.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/depthwise_conv_common.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/depthwise_conv.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/batch_to_space_nd.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/div.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/elu.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/hard_swish.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/l2_pool_2d.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/leaky_relu.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/shape.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/space_to_batch_nd.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/split_v.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/squeeze.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/transpose_conv.cc	./edge-impulse-sdk/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc ./edge-impulse-sdk/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc ./edge-impulse-sdk/tensorflow/lite/c/common.c ./edge-impulse-sdk/tensorflow/lite/core/api/error_reporter.cc ./edge-impulse-sdk/tensorflow/lite/core/api/flatbuffer_conversions.cc ./edge-impulse-sdk/tensorflow/lite/core/api/op_resolver.cc ./edge-impulse-sdk/tensorflow/lite/core/api/tensor_utils.cc ./edge-impulse-sdk/tensorflow/lite/kernels/internal/quantization_util.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/kernel_util_micro.cc ./edge-impulse-sdk/tensorflow/lite/micro/kernels/kernel_runner.cc ./edge-impulse-sdk/tensorflow/lite/kernels/kernel_util_lite.cc ./edge-impulse-sdk/tensorflow/lite/micro/schema_utils.cc ./edge-impulse-sdk/tensorflow/lite/micro/system_setup.cc
SRCS += \
CodeWriter.cc TFLiteMicroBase.cc FirmwareKernels.cc Converter.cc Main.cc

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))))

CXXFLAGS += -std=c++11 -DEI_PORTING_POSIX -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON -I. -I./edge-impulse-sdk
CCFLAGS +=  -std=c11   -DEI_PORTING_POSIX -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON -I. -I./edge-impulse-sdk

LDFLAGS +=  -lm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

tflite_fw : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

all: tflite_fw

clean:
	-$(RM) $(OBJS)
	-$(RM) tflite_fw
	-$(RM) tflite_fw_c_code.c
