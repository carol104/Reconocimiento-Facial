"""Estados temporales de acceso y relé con cierre al terminar la apertura."""
import secrets
import time


class Reto:
    """Reto geométrico auxiliar; no sustituye PAD ni prueba profundidad."""
    def __init__(self, seconds=15):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.steps = None
        self.index = 0
        self.hits = 0
        self.started = None
        self.hold_started = None

    def iniciar(self, now):
        sides = ["IZQUIERDA", "DERECHA"]
        secrets.SystemRandom().shuffle(sides)
        self.steps = ["CENTRO", sides[0], "CENTRO", sides[1], "CENTRO"]
        self.started = now

    @property
    def instruccion(self):
        return self.steps[self.index] if self.steps and self.index < len(self.steps) else "CENTRO"

    def actualizar(self, face, now):
        if self.started is None or now - self.started > self.seconds:
            return "caducado"
        # YuNet: ojo derecho (4,5), ojo izquierdo (6,7), nariz (8,9).
        eye_x = sorted((float(face[4]), float(face[6])))
        span = eye_x[1] - eye_x[0]
        if span < 10:
            return "pendiente"
        offset = (float(face[8]) - sum(eye_x) / 2) / span
        pose = "IZQUIERDA" if offset < -.18 else "DERECHA" if offset > .18 else "CENTRO"
        if pose != self.instruccion:
            self.hits, self.hold_started = 0, None
            return "pendiente"
        if self.hold_started is None:
            self.hold_started = now
        self.hits += 1
        if self.hits >= 3 and now - self.hold_started >= .25:
            self.index += 1
            self.hits, self.hold_started = 0, None
        return "completo" if self.index == len(self.steps) else "pendiente"


class Acceso:
    def __init__(self, cfg):
        self.frames = cfg["frames_confirmacion"]
        self.cooldown = cfg["rele"]["enfriamiento"]
        self.reto = Reto(cfg["reto_segundos"])
        self.next_allowed = 0
        self.reset()

    def reset(self):
        self.person = None
        self.count = 0
        self.last_seen = None
        self.reto.reset()

    def actualizar(self, person, live, face, now):
        if now < self.next_allowed:
            self.reset()
            return False, "Enfriamiento"
        if person is None:
            self.reset()
            return False, "Esperando rostro..."
        if not live:
            self.reset()
            return False, "Acceso denegado: prueba de vida"
        if person == "__desconocido__":
            self.reset()
            return False, "Acceso denegado: persona no autorizada"
        if person != self.person or (self.last_seen is not None and now - self.last_seen > 2):
            self.reset()
            self.person = person
        self.last_seen = now
        self.count += 1
        if self.count < self.frames:
            return False, "Confirmando identidad"
        if self.reto.steps is None:
            self.reto.iniciar(now)
        state = self.reto.actualizar(face, now)
        if state == "caducado":
            self.reset()
            return False, "Reto caducado"
        if state == "completo":
            self.next_allowed = now + self.cooldown
            self.reset()
            return True, "Acceso concedido"
        return False, f"Mueva el rostro: {self.reto.instruccion}"


class Puerta:
    def __init__(self, cfg, hardware=False):
        self.cfg, self.hardware, self.device = cfg, hardware, None

    def __enter__(self):
        if self.hardware:
            from gpiozero import OutputDevice
            self.device = OutputDevice(self.cfg["pin_bcm"],
                active_high=self.cfg["activo_alto"], initial_value=False)
        return self

    def abrir(self):
        if not self.hardware:
            print("SIMULACIÓN: se abriría la puerta.")
            return
        try:
            self.device.on()
            time.sleep(self.cfg["segundos"])
        finally:
            self.device.off()

    def __exit__(self, *args):
        if self.device is not None:
            try:
                self.device.off()
            finally:
                self.device.close()


def verificar_hardware(cfg, artifact, pad_ready):
    if not cfg["despliegue_validado"]:
        raise ValueError("GPIO bloqueado: falta validar identidad, PAD y montaje; despliegue_validado=false.")
    if not pad_ready:
        raise ValueError("GPIO bloqueado: falta un modelo antispoofing entrenado y cargado.")
    if not artifact["evaluacion_basica_aprobada"]:
        raise ValueError("GPIO bloqueado: falta superar validación/test con desconocidos y las cuatro personas.")
