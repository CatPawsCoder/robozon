/*
 * Контроллер 3-позиционного накопителя-лотка (tilt-tray) для ПАК трека 3.
 * Платформа: ESP8266 (NodeMCU). 3 сервопривода: YAW, TILT, FLAP.
 *
 * Реализует ту же логику, что и цифровой двойник sim/conveyor_sim.py:
 *   приём (yaw=0, tilt=приём, флап закрыт) -> поворот на угол зоны ->
 *   опрокидывание (tilt=DUMP, флап открыт) -> возврат.
 *
 * Приём категории двумя способами (связка с частью классификации):
 *   1) Wi-Fi:  HTTP GET  /sort?cat=B|C|D
 *   2) Serial: символ 'B' / 'C' / 'D' (115200)
 *
 * Углы синхронизированы с константами симуляции:
 *   YAW: B=0, C=+55, D=-55 ; TILT: приём=0, опрокидывание=34.
 *
 * ВНИМАНИЕ по питанию: сервоприводы MG996R питать от отдельного БП 5-6 В
 * (>=3 A), землю объединить с ESP8266. От USB сервы не питать.
 */
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Servo.h>

// ---- Wi-Fi (или режим точки доступа, см. USE_AP) ----
#define USE_AP 1                     // 1 = ESP поднимает свою точку доступа
const char* WIFI_SSID = "robozone";
const char* WIFI_PASS = "robozone802";

// ---- пины сервоприводов ----
const int PIN_YAW  = D1;             // GPIO5
const int PIN_TILT = D2;             // GPIO4
const int PIN_FLAP = D5;             // GPIO14

// ---- углы (в градусах серво 0..180; нейтраль 90 = 0° механики) ----
const int YAW_NEUTRAL = 90;          // калибровать под свою механику
const int YAW_C = YAW_NEUTRAL + 55;  // категория C -> +55
const int YAW_D = YAW_NEUTRAL - 55;  // категория D -> -55
const int TILT_RECEIVE = 80;         // приём (лёгкий контруклон)
const int TILT_DUMP    = 80 + 34;    // опрокидывание +34
const int FLAP_CLOSED  = 70;
const int FLAP_OPEN    = 160;

// ---- тайминги цикла, мс ----
const int T_SETTLE = 500;            // осадка товара в ложе
const int T_ROTATE = 600;            // поворот YAW
const int T_DUMP   = 900;            // сход товара
const int T_RETURN = 700;            // возврат в приём

Servo sYaw, sTilt, sFlap;
ESP8266WebServer server(80);

void moveTo(Servo& s, int a) { s.write(constrain(a, 0, 180)); }

void toReceive() {
  moveTo(sFlap, FLAP_CLOSED);
  moveTo(sTilt, TILT_RECEIVE);
  moveTo(sYaw, YAW_NEUTRAL);
}

// полный цикл сортировки по категории
void sortCycle(char cat) {
  int yaw = YAW_NEUTRAL;
  if (cat == 'C') yaw = YAW_C;
  else if (cat == 'D') yaw = YAW_D;      // 'B' -> нейтраль (прямо)

  delay(T_SETTLE);                        // приём и осадка
  moveTo(sYaw, yaw);   delay(T_ROTATE);   // поворот на зону
  moveTo(sTilt, TILT_DUMP);
  moveTo(sFlap, FLAP_OPEN);  delay(T_DUMP);  // опрокидывание + флап
  toReceive();         delay(T_RETURN);   // возврат
}

void handleSort() {
  String c = server.arg("cat"); c.toUpperCase();
  if (c == "B" || c == "C" || c == "D") {
    Serial.printf("HTTP sort -> %s\n", c.c_str());
    sortCycle(c[0]);
    server.send(200, "text/plain", "ok " + c);
  } else {
    server.send(400, "text/plain", "cat must be B|C|D");
  }
}

void setup() {
  Serial.begin(115200);
  sYaw.attach(PIN_YAW); sTilt.attach(PIN_TILT); sFlap.attach(PIN_FLAP);
  toReceive();

#if USE_AP
  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  Serial.print("AP IP: "); Serial.println(WiFi.softAPIP());
#else
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print('.'); }
  Serial.print("\nSTA IP: "); Serial.println(WiFi.localIP());
#endif

  server.on("/sort", handleSort);
  server.on("/", []() { server.send(200, "text/plain",
            "tray_controller ready. GET /sort?cat=B|C|D"); });
  server.begin();
  Serial.println("HTTP server up. Serial: send B/C/D.");
}

void loop() {
  server.handleClient();
  if (Serial.available()) {                 // Serial-фолбэк
    char c = toupper(Serial.read());
    if (c == 'B' || c == 'C' || c == 'D') {
      Serial.printf("Serial sort -> %c\n", c);
      sortCycle(c);
    }
  }
}
