

// This is a standard TensorFlow Lite model file that has been converted into a
// C data array, so it can be easily compiled into a binary for devices that
// don't have a file system. It was created using the command:
// xxd -i sine_model.tflite > sine_model_data.cc

#ifndef DATA_H_
#define DATA_H_

extern const unsigned char mi_data[];
extern const int mi_data_len;

#endif 

extern const float test[30][4];
extern const float labels[30][3];
