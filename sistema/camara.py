import json
from pathlib import Path
import time
import cv2
from sistema.vision import Vision, PAD
from sistema.entrenamiento import cargar_artefacto, identificar
from sistema.acceso import Acceso, Puerta, verificar_hardware


def ejecutar(cfg, algoritmo, camera=0, hardware=False, preview=True):
    artifact = cargar_artefacto(algoritmo, cfg)
    human_artifact = None
    if "fuentes" in cfg:
        from sistema.humanos import cargar_humanos, es_humano
        human_artifact = cargar_humanos(algoritmo, cfg)
    vision, pad = Vision(cfg), PAD(cfg["pad"])
    if hardware:
        verificar_hardware(cfg, artifact, pad.net is not None or pad.local is not None)
    print("Modo GPIO" if hardware else "Modo simulación (sin GPIO)")
    if pad.net is None and pad.local is None:
        print("Falta PAD: todos los accesos serán denegados, incluso en simulación.")
    gate = Acceso(cfg)
    capture = cv2.VideoCapture(camera)
    log_path = Path(cfg["artefactos"]) / "accesos.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_message = None
    try:
        if not capture.isOpened():
            raise RuntimeError("No se pudo abrir la cámara. Comprueba el índice y V4L2.")
        with Puerta(cfg["rele"], hardware) as door:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Se perdió la cámara; acceso detenido.")
                person, face, live = None, None, False
                try:
                    if human_artifact is not None and not es_humano(human_artifact, frame):
                        raise ValueError("El clasificador humano/objeto rechazó la imagen.")
                    vector, crop, face = vision.extraer(frame)
                    person, _, _ = identificar(artifact, vector, crop)
                    live, _ = pad.evaluar(frame, face)
                except ValueError:
                    # Ningún rostro, varios rostros o salida inválida: se cancela la sesión.
                    pass
                allowed, message = gate.actualizar(person, live, face, time.monotonic())
                if message != last_message:
                    print(message)
                    event = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "algoritmo": algoritmo, "evento": message,
                             "persona": person, "gpio": hardware}
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                    last_message = message
                if allowed:
                    door.abrir()
                    # El bloqueo del relé no cuenta como tiempo de enfriamiento.
                    gate.next_allowed = time.monotonic() + cfg["rele"]["enfriamiento"]
                if preview:
                    cv2.putText(frame, message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    cv2.imshow("Acceso facial - Q para salir", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()
