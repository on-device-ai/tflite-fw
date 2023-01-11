#include "CodeWriter.h"

#include <ctime>
#include <iomanip>

#include "tensorflow/lite/c/common.h"

tflite_firmware::CodeWriter::CodeWriter(std::ostream& out)
    : out_(out) {
	// Setup stream: Print booleans as string:
	out_ << std::boolalpha;
	// Print floats with precision that is sufficient for exact back-conversion:
	out_ << std::setprecision(std::numeric_limits<double>::max_digits10);

	out_ << "// This file is generated. Do not edit.\n";
	{
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		out_ << "// Generated on: " << std::put_time(&tm, "%d.%m.%Y %H:%M:%S") << "\n";
	}
}

void tflite_firmware::CodeWriter::writeIntArray(const TfLiteIntArray& arr,
                                      const std::string &ndim_name,
									  const std::string &dims_name) {
	out_ << "const int32_t " << ndim_name << " = " << arr.size << ";\n";
	out_ << "const int32_t " << dims_name << "[] =  { ";
	writeIntArrayData(arr);
	out_ << " };\n";
}

void tflite_firmware::CodeWriter::writeIntArrayData(const TfLiteIntArray& arr) {
	if (arr.size > 0) {
		out_ << arr.data[0];
		for (int i = 1; i < arr.size; i++) {
			out_ << ',' << arr.data[i];
		}
	}
}

// outputting int8_t as a character is not what we intend here, we want to see
// the value, so we introduce printT
template <class T, class printT>
static void dump_tensor_contents(std::ostream& out_, const TfLiteTensor& t,
                                 const std::string& tname,
                                 const std::string& name) {
	if (t.dims->size == 0) {  // special case 0 dimensions, we output an array to
                              // avoid distinction from >0 dimension at every use
		out_ << "const " << tname << " " << name << "[1] = { " << (printT)(tflite::GetTensorData<T>(&t)[0]) << " };\n";
		return;
	}

	uint32_t alignment = t.bytes >= 8 ? 8 : t.bytes >= 4 ? 4 : 2;

	// For packed formats the numer of serialized data items may not
	// necessarily match the nominal dimensions of the tensor.
	// We need to ensure this case is handled correctly.
	size_t nominal_elts = 1;
	for (int i = 0; i < t.dims->size; ++i) {
		nominal_elts *= t.dims->data[i];
	}

	size_t serialized_elts = t.bytes / sizeof(T);

	out_ << "const " << tname << " " << name << "[";

	out_ << t.dims->data[0];
	for (int i = 1; i < t.dims->size; ++i) 
		out_ << '*' << t.dims->data[i];
	out_ << "] = { ";
	if (t.dims->size == 1 || serialized_elts != nominal_elts) {
		// one dimension/packed: 10 per line of data
		for (int i = 0; i < serialized_elts; ++i) {
			if (i % 10 == 0) out_ << "\n    ";
			out_ << (printT)(tflite::GetTensorData<T>(&t)[i]) << ", ";
		}
		out_ << "\n};\n";
	} else if (t.dims->size == 2) {
		// two dimensions: Inner dimension is one line
		for (int i = 0; i < t.dims->data[0]; ++i) {
			out_ << "\n    ";
			for (int j = 0; j < t.dims->data[1]; ++j)
				out_ << (printT)(tflite::GetTensorData<T>(&t)[i * t.dims->data[1] + j]) << ", ";
		}
		out_ << "\n};\n";
	} else {
		// More dimensions: Inner two dimensions per line (space between two
		// middle elements)
		int outer_dim = t.dims->data[0];
		int middle_dim = t.dims->data[t.dims->size - 2];
		int inner_dim = t.dims->data[t.dims->size - 1];
		for (int i = 1; i < t.dims->size - 2; ++i)
			outer_dim *= t.dims->data[i];
		for (int i = 0; i < outer_dim; ++i) {
			// output a meaningful index for this line
			uint32_t idx = i;
			std::string indexstr = "[][]";
			for (int32_t j = t.dims->size - 3; j >= 0; --j) {
				uint32_t idx_j = idx % t.dims->data[j];
				indexstr = "[" + std::to_string(idx_j) + "]" + indexstr;
				idx /= t.dims->data[j];
			}
			out_ << "\n  /* " << indexstr << " */ ";
			for (int j = 0; j < middle_dim; ++j) {
				for (int k = 0; k < inner_dim; ++k)
					out_ << (printT)(tflite::GetTensorData<T>(&t)[(i * middle_dim + j) * inner_dim + k]) << ",";
				out_ << " ";  // separator between middle indices
			}
		}
		out_ << "\n};\n";
	}
}

#define DUMP_TENSOR2(TfType, CType, PrintType)                     \
  case TfType:                                                     \
    dump_tensor_contents<CType, PrintType>(out_, t, #CType, name); \
    break

void tflite_firmware::CodeWriter::writeTensor(const TfLiteTensor& t,
                                                   const std::string& name) {
	switch (t.type) {
		DUMP_TENSOR2(kTfLiteFloat32, float, float);
        DUMP_TENSOR2(kTfLiteInt32, int32_t, int32_t);
        DUMP_TENSOR2(kTfLiteInt8, int8_t, int32_t);
		default:
			break;
	}
}

void tflite_firmware::CodeWriter::writeQuantization(const TfLiteQuantization& q,
                                                         const std::string& name) {
    if (q.type == kTfLiteAffineQuantization) {
        auto aq = (TfLiteAffineQuantization const*)q.params;
        
        if( aq->scale->size > 0 ) {
            out_ << "const float " << name << "_scale[" <<  aq->scale->size << "] = { ";
            out_ << aq->scale->data[0];
            for (int i = 1; i < aq->scale->size; i++) {
                out_ << aq->scale->data[i] << ", ";
            }
        }
        out_ << " };\n";
        
        if( aq->zero_point->size > 0 ) {
            out_ << "const int32_t " << name << "_zero[" <<  aq->zero_point->size << "] = { ";
            out_ << aq->zero_point->data[0]; 
            for (int i = 1; i < aq->zero_point->size; i++) {
                out_ << aq->zero_point->data[i] << ", ";
            }   
        }
        out_ << " };\n";
    }
}
