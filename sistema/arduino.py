"""Módulo de comunicación serial con Arduino para control de acceso."""
import time
import serial
import serial.tools.list_ports


class ArduinoControlador:
    """Controlador que envía estados Verdadero (1) o Falso (0) a un Arduino."""

    def __init__(self, cfg=None):
        self.serial_conn = None
        self.conectado = False
        self.ultimo_estado = None
        self.puerto = None
        self.baudios = 9600

        cfg_arduino = (cfg or {}).get("esp32", (cfg or {}).get("arduino", {}))
        puerto_configurado = cfg_arduino.get("puerto", None)
        self.baudios = cfg_arduino.get("baudios", 9600)
        activo = cfg_arduino.get("activo", True)

        if not activo:
            print("[Arduino] Comunicación desactivada en la configuración.")
            return

        # Lista de puertos a intentar: primero el configurado, luego autodetección
        puertos_a_probar = []
        if puerto_configurado:
            puertos_a_probar.append(puerto_configurado)

        try:
            disponibles = list(serial.tools.list_ports.comports())
            for p in disponibles:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                # Priorizar dispositivos identificados como Arduino o convertidores serial comunes
                if any(k in desc or k in hwid for k in ("arduino", "ch340", "usb-serial", "ftdi", "cp210", "ttyacm", "ttyusb")):
                    puertos_a_probar.insert(1 if puerto_configurado else 0, p.device)
                elif p.device not in puertos_a_probar:
                    puertos_a_probar.append(p.device)
        except Exception:
            pass

        for port in dict.fromkeys(puertos_a_probar):
            try:
                self.serial_conn = serial.Serial(port, self.baudios, timeout=1)
                time.sleep(1.5)  # Tiempo de inicialización tras reset serial de Arduino
                self.conectado = True
                self.puerto = port
                print(f"[Arduino] Conectado exitosamente en {port} a {self.baudios} baudios.")
                break
            except Exception:
                self.serial_conn = None

        if not self.conectado:
            print("[Arduino] Sin Arduino físico conectado. Modo SIMULACIÓN de Arduino activo.")

    def enviar_estado(self, estado_verdadero):
        """
        Envía 1 (Verdadero) o 0 (Falso) a Arduino cuando cambia el estado.
        estado_verdadero: bool (True si al menos una persona está autorizada, False si ninguna)
        """
        if estado_verdadero != self.ultimo_estado:
            self.ultimo_estado = estado_verdadero
            caracter = b'1\n' if estado_verdadero else b'0\n'
            texto = "VERDADERO (1)" if estado_verdadero else "FALSO (0)"

            if self.conectado and self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.write(caracter)
                    self.serial_conn.flush()
                    print(f"[Arduino Hardware ({self.puerto})] Enviado: {texto}")
                except Exception as exc:
                    print(f"[Arduino Error] Fallo al enviar a {self.puerto}: {exc}")
            else:
                print(f"[Arduino Simulado] Señal: {texto}")

    def cerrar(self):
        """Cierra el puerto serial asegurando apagar cualquier señal."""
        if self.conectado and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(b'0\n')
                self.serial_conn.flush()
                self.serial_conn.close()
            except Exception:
                pass
            self.conectado = False
