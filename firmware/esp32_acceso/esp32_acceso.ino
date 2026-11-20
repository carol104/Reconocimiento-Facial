// ESP32 por USB serial: compatible con el protocolo existente 1\n / 0\n.
// Ajusta el pin y la polaridad al módulo de relé de tu montaje.
constexpr int PIN_RELE = 23;
constexpr bool ACTIVO_ALTO = true;

void escribirEstado(bool activo) {
  digitalWrite(PIN_RELE, activo == ACTIVO_ALTO ? HIGH : LOW);
}

void setup() {
  pinMode(PIN_RELE, OUTPUT);
  escribirEstado(false);
  Serial.begin(115200);
}

void loop() {
  static char comando = 0;
  static bool invalido = false;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (!invalido && (comando == '1' || comando == '0')) {
        escribirEstado(comando == '1');
      }
      comando = 0;
      invalido = false;
    } else if (c != '\r') {
      if (comando != 0 || (c != '0' && c != '1')) invalido = true;
      comando = c;
    }
  }
}
