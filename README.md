# TensorFlow Lite for Firmware  
[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) ([TFLite Micro](https://github.com/tensorflow/tflite-micro)) is developed using the C++11 programming language. However, in the scenario of firmware development, the C programming language is used more.
This project attempts to convert the TFLite Micro model into C code. Allow machine learning to be applied in more firmware environments (such as more types of microcontrollers, BIOS of personal computers).  
  
> Concepts for the project have previously been posted on the [Edge AI Taiwan](https://www.facebook.com/groups/edgeaitw/) community (which is a Facebook group) and on SIG Micro's [Gitter chat channel](https://gitter.im/tensorflow/sig-micro).  
  
### Build, execute, and validate
  
* Buid the project under Linux :  
`make`  
  
* Execute after build :  
`./codegen`  
It will convert the quantized model of TFLite Micro's "[Hello World Example](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world)" into the C source code.  
  
* Modify the converted code (in the tflite_fw_c_code.c file), and add the main program:  
  
  ```  
void main() {
    float input_data = 1.57f;
    float output_data;
    
    tensor0_data[0] = input_data / quant0_scale[0] + quant0_zero[0];
    model_invoke();
    output_data = (tensor9_data[0] - quant9_zero[0]) * quant9_scale[0];
    printf("%1.5f\n",output_data);
}
```  
  
* Compile with the following command:  
`gcc tflite_fw_c_code.c -o test_main`  
* Then execute the verification program and its execution result is as follows:  
**$** ./test_main  
1.04206  