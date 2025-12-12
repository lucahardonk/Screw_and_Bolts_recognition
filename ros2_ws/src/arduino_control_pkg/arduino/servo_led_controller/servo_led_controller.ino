/*
===============================================================
  SERIAL COMMAND USAGE
===============================================================

Send commands through Serial (115200 baud):

1) CONTROL A SERVO (INPUT LIMITED TO 0–180)
------------------------------------------------
  SERVO,<id>,<angle>

Where:
  <id>     = 1 or 2
  <angle>  = 0–180 degrees (INPUT LIMIT)

Examples:
  SERVO,1,0       → moves clamp servo fully open
  SERVO,1,90      → mid-angle
  SERVO,2,180     → wrist servo to max allowed angle

⚠ IMPORTANT:
Servo hardware rotates 270°, but SERIAL INPUT is limited to 180°.
Value is internally remapped to the full 270° hardware range.

2) SET ONE NEOPIXEL LED COLOR
------------------------------------------------
  LED,<index>,<R>,<G>,<B>

Example:
  LED,3,255,0,0   → LED #3 becomes RED
===============================================================
*/

#include <Servo.h>
#include <Adafruit_NeoPixel.h>

// ===============================
// CONFIGURATION
// ===============================
static const uint8_t CLAMP_SERVO_PIN = 9; // id 1
static const uint8_t WRIST_SERVO_PIN = 10;  // id 2

static const uint8_t NEOPIXEL_PIN = 6;
static const uint8_t NUM_LEDS     = 8;

// Servo hardware characteristics (your 270° servos)
static const int SERVO_MIN_DEG = 0;
static const int SERVO_MAX_DEG = 270;

// Serial input range (user commands)
static const int MAX_SERIAL_ANGLE = 180;  // 0–180 limit

// Servo library range
static const int SERVO_LIB_MIN = 0;
static const int SERVO_LIB_MAX = 180;

// ===============================
// GLOBAL OBJECTS
// ===============================
Servo clamp_servo; // id 1
Servo wrist_servo; // id 2

Adafruit_NeoPixel strip(NUM_LEDS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

String inputLine = "";

// ===============================
// HELPER FUNCTIONS
// ===============================

// Maps 0–180 (serial input) → 0–270 (real servo) → 0–180 (library)
int mapServoAngle(int inputAngle) {
  inputAngle = constrain(inputAngle, 0, 180);

  // 1) Map 0–180 → 40–230 real degrees
  int realAngle = map(inputAngle,
                      0, 180,
                      40, 230);

  // 2) Convert real 40–230 → library 0–180
  int libAngle = map(realAngle,
                     0, 270,
                     0, 180);

  return libAngle;
}



void setServo(uint8_t id, int serialAngle) {
  int mappedAngle = mapServoAngle(serialAngle);

  switch (id) {
    case 1:
      clamp_servo.write(mappedAngle);
      Serial.println("OK: clamp_servo");
      break;

    case 2:
      wrist_servo.write(mappedAngle);
      Serial.println("OK: wrist_servo");
      break;

    default:
      Serial.println("ERR: Invalid SERVO ID");
      break;
  }
}

void setLedColor(uint8_t index, uint8_t r, uint8_t g, uint8_t b) {
  if (index >= NUM_LEDS) {
    Serial.println("ERR: LED INDEX");
    return;
  }

  strip.setPixelColor(index, strip.Color(r, g, b));
  strip.show();

  Serial.print("OK: LED ");
  Serial.println(index);
}

// ===============================
// COMMAND PARSING
// ===============================

void handleServoCommand(const String &cmd) {
  int c1 = cmd.indexOf(',');
  int c2 = cmd.indexOf(',', c1 + 1);

  if (c1 < 0 || c2 < 0) {
    Serial.println("ERR: SERVO FORMAT");
    return;
  }

  int id      = cmd.substring(c1 + 1, c2).toInt();
  int angle   = cmd.substring(c2 + 1).toInt();

  setServo(id, angle);
}

void handleLedCommand(const String &cmd) {
  int c1 = cmd.indexOf(',');
  int c2 = cmd.indexOf(',', c1 + 1);
  int c3 = cmd.indexOf(',', c2 + 1);
  int c4 = cmd.indexOf(',', c3 + 1);

  if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0) {
    Serial.println("ERR: LED FORMAT");
    return;
  }

  uint8_t index = cmd.substring(c1 + 1, c2).toInt();
  uint8_t r     = constrain(cmd.substring(c2 + 1, c3).toInt(), 0, 255);
  uint8_t g     = constrain(cmd.substring(c3 + 1, c4).toInt(), 0, 255);
  uint8_t b     = constrain(cmd.substring(c4 + 1).toInt(),     0, 255);

  setLedColor(index, r, g, b);
}

void handleCommand(const String &cmd) {
  if (cmd.length() == 0) return;

  if (cmd.startsWith("SERVO")) 
      handleServoCommand(cmd);

  else if (cmd.startsWith("LED")) 
      handleLedCommand(cmd);

  else 
      Serial.println("ERR: UNKNOWN CMD");
}

// ===============================
// SETUP & MAIN LOOP
// ===============================

void setup() {
  Serial.begin(115200);

  clamp_servo.attach(CLAMP_SERVO_PIN);
  wrist_servo.attach(WRIST_SERVO_PIN);

  setServo(1, 0);
  setServo(2, 90);

  strip.begin();
  strip.show();

  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      handleCommand(inputLine);
      inputLine = "";
    }
    else if (c != '\r') {
      inputLine += c;
    }
  }
}
