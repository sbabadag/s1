#include <TensorFlowLite_ESP32.h>
#include <I2S.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// Audio settings
#define SAMPLE_RATE 16000
#define SAMPLE_BITS 16
#define I2S_BUFFER_SIZE 1024
#define AUDIO_LENGTH (16000 * 3)  // 3 seconds of audio at 16kHz

// TensorFlow Lite interpreter
tflite::MicroInterpreter* interpreter = nullptr;

// Input and output tensors
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 100 * 1024;  // Increased from 60KB to 100KB
uint8_t tensor_arena[kTensorArenaSize];

// Touch Switch variables
int threshold = 1500;  // Adjust if not responding properly
bool touch1detected = false;

// Buffer for audio processing moved to global scope
int16_t audio_buffer[I2S_BUFFER_SIZE];

// Callback function for touch switch
void gotTouch1() {
  touch1detected = true;
}

// Function declarations
void generate_wav_header(uint8_t *wav_header, uint32_t wav_size, uint32_t sample_rate);
void record_wav();

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Serial initialized");

  // Initialize I2S
  I2S.setAllPins(-1, 42, 41, -1, -1);
  Serial.println("I2S pins set");
  if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
    Serial.println("Failed to initialize I2S!");
    while (1);
  }
  Serial.println("I2S initialized successfully");

  // Attach touch switch to interrupt handler
  touchAttachInterrupt(T1, gotTouch1, threshold);
  Serial.println("Touch interrupt attached");

  // Initialize TensorFlow Lite
  const tflite::Model* model = tflite::GetModel(g_model);
    Serial.println("model loaded");

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }
  Serial.println("TensorFlow Lite model loaded");

  // Set up the resolver
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  Serial.println("TensorFlow Lite resolver set up");

  // Initialize the interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;
  Serial.println("TensorFlow Lite interpreter initialized");

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }
  Serial.println("TensorFlow Lite tensors allocated");

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TensorFlow Lite input and output tensors obtained");

  Serial.println("Setup complete!");
}

void loop() {
  if (touch1detected) {
    Serial.println("Touch detected, starting wake word detection...");

    // Check available heap memory
    Serial.print("Free heap before processing: ");
    Serial.println(ESP.getFreeHeap());

    // Process audio in chunks
    for (int offset = 0; offset < AUDIO_LENGTH; offset += I2S_BUFFER_SIZE) {
      Serial.println("Reading audio data from I2S");
      // Read audio data directly from I2S
      size_t bytes_read = I2S.readBytes((char*)audio_buffer, sizeof(audio_buffer));

      if (bytes_read == 0) {
        Serial.println("Failed to read I2S data");
        continue;
      }
      Serial.print("Bytes read: ");
      Serial.println(bytes_read);

      // Preprocess audio data
      Serial.println("Preprocessing audio data");
      for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
        input->data.int8[i] = static_cast<int8_t>(audio_buffer[i] / 32768.0 * 128);  // Normalize to [-128, 127]
      }

      // Run inference
      Serial.println("Running TensorFlow Lite inference");
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        continue;
      }
      Serial.println("Inference completed");

      // Process results
      int8_t* output_data = output->data.int8;

      // Check for wake word "hey selo"
      if (output_data[0] > 128) {
        Serial.println("Wake word detected!");

        // Wait a moment for the command
        delay(500);
        Serial.println("Waiting for command audio");

        // Read command audio
        size_t cmd_bytes_read = I2S.readBytes((char*)audio_buffer, sizeof(audio_buffer));
        Serial.print("Command bytes read: ");
        Serial.println(cmd_bytes_read);
        if (cmd_bytes_read > 0) {
          // Process command audio
          Serial.println("Preprocessing command audio");
          for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
            input->data.int8[i] = static_cast<int8_t>(audio_buffer[i] / 32768.0 * 128);
          }

          Serial.println("Running inference on command audio");
          invoke_status = interpreter->Invoke();
          if (invoke_status == kTfLiteOk) {
            // Check for commands
            if (output_data[1] > 128) {
              Serial.println("Command: aç");
            } else if (output_data[2] > 128) {
              Serial.println("Command: kapat");
            } else {
              Serial.println("Unknown command");
            }
          } else {
            Serial.println("Invoke failed on command audio!");
          }
        }
        break;  // Exit the processing loop after detecting wake word
      }

      // Check available heap memory after processing
      Serial.print("Free heap after processing chunk: ");
      Serial.println(ESP.getFreeHeap());
    }

    touch1detected = false;
    Serial.println("Touch processing complete");

    // Check available heap memory after processing
    Serial.print("Free heap after processing: ");
    Serial.println(ESP.getFreeHeap());
  }

  delay(100);
}

// Removed unused functions