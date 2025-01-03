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
#define AUDIO_LENGTH 16000 * 3  // 3 seconds of audio at 16kHz
// TensorFlow Lite interpreter
tflite::MicroInterpreter* interpreter = nullptr;

// Input and output tensors
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int threshold = 1500;  // Adjust if not responding properly
bool touch1detected = false;

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
  if (!I2S.begin(I2S_LEFT_JUSTIFIED_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
    Serial.println("Failed to initialize I2S!");
    while (1);
  }
  Serial.println("I2S initialized successfully");

  // Attach touch switch to interrupt handler
  touchAttachInterrupt(T1, gotTouch1, threshold);
  Serial.println("Touch interrupt attached");

  // Initialize TensorFlow Lite
  const tflite::Model* model = tflite::GetModel(g_model);
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
  interpreter->AllocateTensors();
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
    
    // Buffer for audio processing
    int16_t audio_buffer[I2S_BUFFER_SIZE];
    size_t bytes_read;
    
    // Process audio in chunks
    for (int offset = 0; offset < AUDIO_LENGTH; offset += I2S_BUFFER_SIZE) {
      Serial.println("Reading audio data from I2S");
      // Read audio data directly from I2S
      bytes_read = I2S.readBytes((char*)audio_buffer, sizeof(audio_buffer));
      
      if (bytes_read == 0) {
        Serial.println("Failed to read I2S data");
        continue;
      }
      Serial.print("Bytes read: ");
      Serial.println(bytes_read);

      // Preprocess audio data
      Serial.println("Preprocessing audio data");
      for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
        input->data.int8[i] = audio_buffer[i] / 32768.0 * 128;  // Normalize to [-128, 127]
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
        bytes_read = I2S.readBytes((char*)audio_buffer, sizeof(audio_buffer));
        Serial.print("Command bytes read: ");
        Serial.println(bytes_read);
        if (bytes_read > 0) {
          // Process command audio
          Serial.println("Preprocessing command audio");
          for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
            input->data.int8[i] = audio_buffer[i] / 32768.0 * 128;
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
    }
    
    touch1detected = false;
    Serial.println("Touch processing complete");
  }
  
  delay(100);
}

void generate_wav_header(uint8_t *wav_header, uint32_t wav_size, uint32_t sample_rate) {
    // WAV header format
    wav_header[0] = 'R';  // RIFF
    wav_header[1] = 'I';
    wav_header[2] = 'F';
    wav_header[3] = 'F';
    uint32_t file_size = wav_size + 36;
    wav_header[4] = (file_size & 0xFF);  // File size
    wav_header[5] = ((file_size >> 8) & 0xFF);
    wav_header[6] = ((file_size >> 16) & 0xFF);
    wav_header[7] = ((file_size >> 24) & 0xFF);
    wav_header[8] = 'W';  // WAVE
    wav_header[9] = 'A';
    wav_header[10] = 'V';
    wav_header[11] = 'E';
    wav_header[12] = 'f';  // fmt
    wav_header[13] = 'm';
    wav_header[14] = 't';
    wav_header[15] = ' ';
    wav_header[16] = 0x10;  // Subchunk size
    wav_header[17] = 0x00;
    wav_header[18] = 0x00;
    wav_header[19] = 0x00;
    wav_header[20] = 0x01;  // PCM
    wav_header[21] = 0x00;
    wav_header[22] = 0x01;  // Mono
    wav_header[23] = 0x00;
    wav_header[24] = (sample_rate & 0xFF);  // Sample rate
    wav_header[25] = ((sample_rate >> 8) & 0xFF);
    wav_header[26] = ((sample_rate >> 16) & 0xFF);
    wav_header[27] = ((sample_rate >> 24) & 0xFF);
    uint32_t byte_rate = sample_rate * 2;
    wav_header[28] = (byte_rate & 0xFF);  // Byte rate
    wav_header[29] = ((byte_rate >> 8) & 0xFF);
    wav_header[30] = ((byte_rate >> 16) & 0xFF);
    wav_header[31] = ((byte_rate >> 24) & 0xFF);
    wav_header[32] = 0x02;  // Block align
    wav_header[33] = 0x00;
    wav_header[34] = 0x10;  // Bits per sample
    wav_header[35] = 0x00;
    wav_header[36] = 'd';  // data
    wav_header[37] = 'a';
    wav_header[38] = 't';
    wav_header[39] = 'a';
    wav_header[40] = (wav_size & 0xFF);  // Data size
    wav_header[41] = ((wav_size >> 8) & 0xFF);
    wav_header[42] = ((wav_size >> 16) & 0xFF);
    wav_header[43] = ((wav_size >> 24) & 0xFF);
}