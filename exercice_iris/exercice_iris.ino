#include <TensorFlowLite.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "data.h"

namespace { 
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 4 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}  



void setup() {
  Serial.begin(115200);
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;


  model = tflite::GetModel(mi_data);

  static tflite::ops::micro::AllOpsResolver resolver;

  // Cree un intérprete para ejecutar el modelo.
  static tflite::MicroInterpreter static_interpreter( model, 
                                                      resolver, 
                                                      tensor_arena,
                                                      kTensorArenaSize, 
                                                      error_reporter);
  interpreter = &static_interpreter;

  // Asignar memoria  para los tensores del modelo.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  // Obtenga punteros a los tensores de entrada y salida del modelo.
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("|-----------------------TensorFlowLite-ESP32-----------------------|");
  Serial.println("|---------------------------IRIS DATA SET--------------------------|");
  Serial.println("|-----------Prediction-ESP32----------|---------Real label---------|");
  
  // Calcule un valor x para alimentar el modelo.
  int samples = 30;
  int features = 4;
  for (int i = 0; i < samples; i++) {
    for (int u = 0; u < features; u++) {
      // Coloque el valor calculado en el tensor de entrada del modelo
      input->data.f[u] = test[i][u];
    }
    
    // estas líneas de código son para comparar con las etiquetas reales
    
    float real1 = labels[i][0];
    float real2 = labels[i][1];
    float real3 = labels[i][2];

    // Ejecute el modelo en esta entrada y verifique que tenga éxito
    TfLiteStatus invoke_status = interpreter->Invoke();


    // Obtenga el valor de salida del tensor
    
    float out_1 = output->data.f[0];
    float out_2 = output->data.f[1];
    float out_3 = output->data.f[2];

    Serial.print("|   ");
    Serial.print(out_1,5);
    Serial.print("     ");
    Serial.print(out_2,5);
    Serial.print("     ");
    Serial.print(out_3,5);
    Serial.print("   |    ");
    Serial.print(real1);
    Serial.print("    ");
    Serial.print(real2);
    Serial.print("    ");
    Serial.print(real3);
    Serial.print("    |\n");
  }

  Serial.println("|-------------------------------------|----------------------------|");
}

void loop() {

}

//by sandro ormeño
