# TensorFlow Lite for Firmware  
[TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) ([TFLite Micro](https://github.com/tensorflow/tflite-micro)) is developed using the C++11 programming language. However, in the scenario of firmware development, the C programming language is used more.  
This project attempts to convert the TFLite Micro model into C code. Allow machine learning to be applied in more firmware environments (such as the BIOS of personal computers).  
  
> Concepts for this project have previously been posted on the [Edge AI Taiwan](https://www.facebook.com/groups/edgeaitw/) community (which is a Facebook group) and on SIG Micro's [Gitter chat channel](https://gitter.im/tensorflow/sig-micro).  
  
### Build, execute, and validate
  
* Buid the project under Linux :  
`make`  
  
* Execute after build :  
`./tflite_fw ./examples/model.tflite`  
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
  
### Verify under UEFI Shell  
Verification can also be performed under [UEFI](https://uefi.org/) Shell. At this time, you need to install the [EDK2](https://github.com/tianocore/edk2) development environment. And I use the [emulator](https://github.com/tianocore/edk2/blob/master/EmulatorPkg/Readme.md) attached to EDK2 as the verification environment.  
In addition, the [edk2-libc](https://github.com/tianocore/edk2-libc) library is required when compiling the verification application, and the completed source code can be downloaded from here: [https://tinyurl.com/2p727w3x](https://tinyurl.com/2p727w3x) . The execution result is as follows:  
  
![230112](https://user-images.githubusercontent.com/44540872/212110306-6ad8e4ce-d07a-48d6-9dcc-7c8c2bfccb62.png)

