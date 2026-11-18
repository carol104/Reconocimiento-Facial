"""Módulo de captura de cámara, guía visual limpia y control de acceso facial.
Diseñado para ser responsive y minimalista en laptop y Raspberry Pi con cámara.
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from sistema.vision import Vision, PAD
from sistema.entrenamiento import cargar_artefacto, identificar
from sistema.acceso import Acceso, Puerta, verificar_hardware


def dibujar_esquinas(frame, x1, y1, x2, y2, color, grosor=3, longitud=35):
    """Dibuja esquinas estilizadas y proporcionales en el recuadro guía de encuadre."""
    # Superior izquierda
    cv2.line(frame, (x1, y1), (x1 + longitud, y1), color, grosor)
    cv2.line(frame, (x1, y1), (x1, y1 + longitud), color, grosor)
    # Superior derecha
    cv2.line(frame, (x2, y1), (x2 - longitud, y1), color, grosor)
    cv2.line(frame, (x2, y1), (x2 - longitud, y1 + longitud), color, grosor)
    # Inferior izquierda
    cv2.line(frame, (x1, y2), (x1 + longitud, y2), color, grosor)
    cv2.line(frame, (x1, y2), (x1, y2 - longitud), color, grosor)
    # Inferior derecha
    cv2.line(frame, (x2, y2), (x2 - longitud, y2), color, grosor)
    cv2.line(frame, (x2, y2), (x2 - longitud, y2 - longitud), color, grosor)


def superponer_panel(frame, y1, y2, color=(16, 16, 16), alpha=0.75):
    """Dibuja una franja semitransparente adaptable para resaltar textos."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, max(0, y1)), (frame.shape[1], min(frame.shape[0], y2)), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def calcular_escala_texto(texto, max_ancho, escala_base, grosor=2):
    """Calcula la escala exacta de fuente para que el texto nunca se corte."""
    escala = escala_base
    while escala > 0.35:
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        if tw <= max_ancho:
            break
        escala -= 0.04
    return max(0.35, escala)


def dibujar_texto_centrado(frame, texto, y_centro, color, escala_base=1.0, grosor=3):
    """Dibuja texto perfectamente centrado horizontalmente en la pantalla."""
    w = frame.shape[1]
    max_w = int(w * 0.92)
    escala = calcular_escala_texto(texto, max_w, escala_base, grosor)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x = max(10, (w - tw) // 2)
    y = int(y_centro + th / 2)
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def ejecutar(cfg, algoritmo, camera=0, hardware=False, preview=True):
    artifact = cargar_artefacto(algoritmo, cfg)
    human_artifact = None
    if "fuentes" in cfg:
        from sistema.humanos import cargar_humanos, es_humano
        human_artifact = cargar_humanos(algoritmo, cfg)
    vision, pad = Vision(cfg), PAD(cfg["pad"])
    if hardware:
        verificar_hardware(cfg, artifact, pad.net is not None or pad.local is not None)

    print("Modo GPIO (Hardware activado)" if hardware else "Modo simulación (sin GPIO)")
    if pad.net is None and pad.local is None:
        if hardware:
            print("Aviso: Falta PAD para hardware.")
        else:
            print("Modo simulación: Reconocimiento facial directo y activo.")

    gate = Acceso(cfg)
    capture = cv2.VideoCapture(camera)

    # Intento de resolución HD adaptable a laptop o Raspberry Pi
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Control de Acceso Facial IA"
    is_fullscreen = False

    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    log_path = Path(cfg["artefactos"]) / "accesos.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_log_message = None

    def registrar_evento(evento, persona=None):
        nonlocal last_log_message
        if evento != last_log_message:
            print(f"[{time.strftime('%H:%M:%S')}] {evento}" + (f" ({persona})" if persona else ""))
            data = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "algoritmo": algoritmo,
                "evento": evento,
                "persona": persona,
                "gpio": hardware
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(data, ensure_ascii=False) + "\n")
            last_log_message = evento

    # Variables de control
    estabilidad_cuadros = 0
    REQUERIDOS_ESTABILIDAD = 3
    resultado_bloqueo_hasta = 0
    resultado_tipo = None  # "CONCEDIDO" o "DENEGADO"
    resultado_mensaje = ""

    # Colores BGR
    AMARILLO = (0, 220, 255)
    VERDE = (0, 255, 0)
    ROJO = (0, 0, 255)

    try:
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera}. Comprueba que esté conectada.")

        with Puerta(cfg["rele"], hardware) as door:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Se perdió la señal de la cámara; acceso detenido.")

                now = time.monotonic()
                h, w = frame.shape[:2]

                # ==============================================================
                # CÁLCULOS RESPONSIVE (Laptop y Raspberry Pi)
                # ==============================================================
                top_h = max(60, int(h * 0.14))
                base_scale_res = max(0.8, min(1.4, w / 850.0))

                # Recuadro central de encuadre guía
                gw = int(w * 0.42)
                gh = int(h * 0.58)
                gx1 = (w - gw) // 2
                gy1 = (h - gh) // 2
                gx2 = gx1 + gw
                gy2 = gy1 + gh
                gcx = (gx1 + gx2) / 2
                gcy = (gy1 + gy2) / 2
                corner_len = max(20, int(gw * 0.14))
                corner_thick = max(2, int(w / 400))
                face_thick = max(2, int(w / 450))

                min_size = min(cfg.get("rostro_minimo_px", 100), int(min(w, h) * 0.20))

                # ==============================================================
                # 1. PANTALLA DE RESULTADO FINAL DETERMINANTE
                # ==============================================================
                if now < resultado_bloqueo_hasta:
                    if preview:
                        color_res = VERDE if resultado_tipo == "CONCEDIDO" else ROJO

                        # Panel superior con el texto final centrado
                        superponer_panel(frame, 0, top_h, color=(12, 12, 12), alpha=0.82)
                        dibujar_texto_centrado(frame, resultado_mensaje, top_h // 2, color_res,
                                               escala_base=base_scale_res, grosor=3)

                        # Recuadro en el color de resultado
                        dibujar_esquinas(frame, gx1, gy1, gx2, gy2, color_res,
                                         grosor=corner_thick + 1, longitud=corner_len)

                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            break
                        elif key in (ord("f"), ord("F")):
                            is_fullscreen = not is_fullscreen
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                                  cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
                    continue

                # ==============================================================
                # 2. SEGUIMIENTO VISUAL: AMARILLO O VERDE SEGÚN POSICIÓN
                # ==============================================================
                if now < gate.next_allowed:
                    # Enfriamiento tras abrir la puerta
                    color_cuadro = AMARILLO
                    cara_detectada = None
                else:
                    vision.detector.setInputSize((w, h))
                    _, faces = vision.detector.detect(frame)

                    color_cuadro = AMARILLO
                    cara_detectada = None
                    rostro_listo = False

                    if faces is None or len(faces) == 0:
                        estabilidad_cuadros = 0
                        gate.reset()
                    elif len(faces) > 1:
                        estabilidad_cuadros = 0
                        gate.reset()
                    else:
                        face = faces[0]
                        cara_detectada = face
                        fx, fy, fw, fh = face[:4]
                        fcx, fcy = fx + fw / 2, fy + fh / 2

                        # Comprobaciones de posición, encuadre y nitidez
                        dx = abs(fcx - gcx)
                        dy = abs(fcy - gcy)

                        cy1, cy2 = max(0, int(fy)), min(h, int(fy + fh))
                        cx1, cx2 = max(0, int(fx)), min(w, int(fx + fw))
                        crop_face = frame[cy1:cy2, cx1:cx2]
                        blur_val = cv2.Laplacian(cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() if crop_face.size > 0 else 0

                        eye_x = sorted((float(face[4]), float(face[6])))
                        span = eye_x[1] - eye_x[0]
                        offset = (float(face[8]) - sum(eye_x) / 2) / span if span > 10 else 0

                        # Validar si está bien posicionado y enfocado dentro del cuadro
                        esta_bien = True
                        if min(fw, fh) < min_size or fw < gw * 0.45 or fw > gw * 1.15 or fh > gh * 1.15:
                            esta_bien = False
                        elif dx > gw * 0.28 or dy > gh * 0.28 or fx < 0 or fy < 0 or fx + fw > w or fy + fh > h:
                            esta_bien = False
                        elif blur_val < 32.0:
                            esta_bien = False
                        elif abs(offset) > 0.22:
                            esta_bien = False

                        if not esta_bien:
                            # Amarillo: no está dentro del cuadro o no se ve bien
                            color_cuadro = AMARILLO
                            estabilidad_cuadros = 0
                        else:
                            # Verde: está bien posicionado y enfocado
                            color_cuadro = VERDE
                            estabilidad_cuadros += 1
                            if estabilidad_cuadros >= REQUERIDOS_ESTABILIDAD:
                                rostro_listo = True

                        # ==============================================================
                        # 3. ESCANEO Y RESPUESTA FINAL DETERMINANTE
                        # ==============================================================
                        if rostro_listo:
                            if human_artifact is not None and not es_humano(human_artifact, frame):
                                color_cuadro = AMARILLO
                            else:
                                try:
                                    aligned = vision.recognizer.alignCrop(frame, face)
                                    vector = vision.recognizer.feature(aligned).flatten()
                                    norm = np.linalg.norm(vector)
                                    if norm >= 1e-8:
                                        vector = vector / norm
                                        crop = cv2.resize(aligned, (64, 64))
                                        tensor = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
                                        person, score, margin = identificar(artifact, vector, tensor)

                                        live = True
                                        if pad.local is not None or pad.net is not None:
                                            live, _ = pad.evaluar(frame, face)

                                        if person in cfg["autorizados"] and live:
                                            # AUTORIZADO: TEXTO EN VERDE CON BIENVENIDO Y NOMBRE
                                            nombre_formateado = person.replace('_', ' ').upper()
                                            resultado_bloqueo_hasta = now + 3.5
                                            resultado_tipo = "CONCEDIDO"
                                            resultado_mensaje = f"BIENVENIDO {nombre_formateado}"
                                            door.abrir()
                                            gate.next_allowed = now + cfg["rele"]["enfriamiento"]
                                            registrar_evento(f"Acceso concedido: {nombre_formateado}", person)
                                            gate.reset()
                                            estabilidad_cuadros = 0
                                        else:
                                            # NO AUTORIZADO: TEXTO EN ROJO CON ACCESO DENEGADO
                                            resultado_bloqueo_hasta = now + 3.0
                                            resultado_tipo = "DENEGADO"
                                            resultado_mensaje = "ACCESO DENEGADO"
                                            registrar_evento("Acceso denegado: persona no autorizada", person or "__desconocido__")
                                            gate.reset()
                                            estabilidad_cuadros = 0
                                except ValueError:
                                    pass

                # ==============================================================
                # 4. RENDERIZADO VISUAL LIMPIO (SIN TEXTOS DE AJUSTE)
                # ==============================================================
                if preview:
                    # Recuadro central de encuadre (Amarillo o Verde)
                    dibujar_esquinas(frame, gx1, gy1, gx2, gy2, color_cuadro,
                                     grosor=corner_thick, longitud=corner_len)

                    # Cuadro alrededor de la cara detectada (Amarillo o Verde)
                    if cara_detectada is not None:
                        fx, fy, fw, fh = cara_detectada[:4].astype(int)
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color_cuadro, face_thick)

                    # Indicador discreto de salida en la esquina inferior
                    cv2.putText(frame, "Q: Salir | F: Pantalla", (16, h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    elif key in (ord("f"), ord("F")):
                        is_fullscreen = not is_fullscreen
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()
