"""Módulo de captura de cámara, guía de alineación/enfoque y control de acceso facial.
Diseñado para ser responsive y adaptarse tanto a laptops como a Raspberry Pi con cámara.
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
    cv2.line(frame, (x2, y1), (x2, y1 + longitud), color, grosor)
    # Inferior izquierda
    cv2.line(frame, (x1, y2), (x1 + longitud, y2), color, grosor)
    cv2.line(frame, (x1, y2), (x1, y2 - longitud), color, grosor)
    # Inferior derecha
    cv2.line(frame, (x2, y2), (x2 - longitud, y2), color, grosor)
    cv2.line(frame, (x2, y2), (x2, y2 - longitud), color, grosor)


def superponer_panel(frame, y1, y2, color=(16, 16, 16), alpha=0.72):
    """Dibuja una franja semitransparente adaptable para textos e instrucciones."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, max(0, y1)), (frame.shape[1], min(frame.shape[0], y2)), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def calcular_escala_texto(texto, max_ancho, escala_base, grosor=2):
    """Calcula la escala exacta de fuente para que el texto nunca se corte en pantallas pequeñas."""
    escala = escala_base
    while escala > 0.35:
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        if tw <= max_ancho:
            break
        escala -= 0.04
    return max(0.35, escala)


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

    # Intento de resolución HD; en Raspberry Pi o webcams que no soporten 720p,
    # OpenCV negociará automáticamente la resolución nativa adecuada (ej. 640x480 o 800x600).
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Control de Acceso Facial IA"
    is_fullscreen = False

    if preview:
        # Ventana responsive: permite redimensionarse y adaptarse a laptop o pantallas de Raspberry Pi
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

    # Variables de control de estabilidad y respuesta
    estabilidad_cuadros = 0
    REQUERIDOS_ESTABILIDAD = 4
    resultado_bloqueo_hasta = 0
    resultado_tipo = None  # "CONCEDIDO" o "DENEGADO"
    resultado_mensaje = ""
    resultado_submensaje = ""

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
                # CÁLCULOS RESPONSIVE (Adaptables a Laptop y Raspberry Pi)
                # ==============================================================
                top_h = max(55, int(h * 0.13))
                bot_h = max(45, int(h * 0.10))
                margin_x = max(16, int(w * 0.03))
                max_text_w = w - (margin_x * 2)

                # Escalas de texto base proporcionadas al ancho
                base_scale_main = max(0.58, min(1.2, w / 950.0))
                base_scale_title = max(0.42, min(0.65, w / 1500.0))
                base_scale_sub = max(0.48, min(0.75, w / 1300.0))

                # Recuadro central de encuadre responsive
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

                # Tamaño mínimo de rostro adaptado a la resolución actual
                min_size = min(cfg.get("rostro_minimo_px", 100), int(min(w, h) * 0.20))

                # ==============================================================
                # 1. PANTALLA DE RESPUESTA DETERMINANTE (VERDE / ROJO)
                # ==============================================================
                if now < resultado_bloqueo_hasta:
                    if preview:
                        color_res = (0, 255, 0) if resultado_tipo == "CONCEDIDO" else (0, 0, 255)

                        # Panel superior
                        superponer_panel(frame, 0, top_h, color=(12, 12, 12), alpha=0.75)
                        title_text = f"SISTEMA DE ACCESO FACIAL IA | Algoritmo: {algoritmo.upper()}"
                        sc_title = calcular_escala_texto(title_text, max_text_w, base_scale_title, grosor=1)
                        cv2.putText(frame, title_text, (margin_x, int(top_h * 0.38)),
                                    cv2.FONT_HERSHEY_SIMPLEX, sc_title, (200, 200, 200), 1, cv2.LINE_AA)

                        sc_res = calcular_escala_texto(resultado_mensaje, max_text_w, base_scale_main * 1.25, grosor=3)
                        cv2.putText(frame, resultado_mensaje, (margin_x, int(top_h * 0.84)),
                                    cv2.FONT_HERSHEY_SIMPLEX, sc_res, color_res, 3, cv2.LINE_AA)

                        # Recuadro guía en color determinante
                        dibujar_esquinas(frame, gx1, gy1, gx2, gy2, color_res, grosor=corner_thick + 1, longitud=corner_len)

                        # Panel inferior
                        superponer_panel(frame, h - bot_h, h, color=(12, 12, 12), alpha=0.75)
                        sc_sub = calcular_escala_texto(resultado_submensaje, max_text_w - 180, base_scale_sub, grosor=1)
                        cv2.putText(frame, resultado_submensaje, (margin_x, h - int(bot_h * 0.38)),
                                    cv2.FONT_HERSHEY_SIMPLEX, sc_sub, (255, 255, 255), 1, cv2.LINE_AA)

                        hint_text = "Q: Salir | F: Pantalla" if w >= 600 else "Q: Salir"
                        cv2.putText(frame, hint_text, (w - int(len(hint_text) * 11 * base_scale_title) - margin_x, h - int(bot_h * 0.38)),
                                    cv2.FONT_HERSHEY_SIMPLEX, base_scale_title, (180, 180, 180), 1, cv2.LINE_AA)

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
                # 2. ENFRIAMIENTO POST-APERTURA
                # ==============================================================
                if now < gate.next_allowed:
                    instruccion_principal = "PUERTA ABIERTA / ESPERE..."
                    instruccion_secundaria = f"Cerrando relé. Siguiente acceso en {int(gate.next_allowed - now) + 1}s"
                    color_guia = (200, 200, 200)
                    cara_detectada = None
                else:
                    # ==============================================================
                    # 3. DETECCIÓN Y EVALUACIÓN DE ROSTRO
                    # ==============================================================
                    vision.detector.setInputSize((w, h))
                    _, faces = vision.detector.detect(frame)

                    instruccion_principal = "POR FAVOR, COLÓQUESE FRENTE A LA CÁMARA"
                    instruccion_secundaria = "Ubique su rostro dentro del recuadro central"
                    color_guia = (220, 220, 220)
                    cara_detectada = None
                    rostro_listo = False

                    if faces is None or len(faces) == 0:
                        estabilidad_cuadros = 0
                        gate.reset()
                    elif len(faces) > 1:
                        estabilidad_cuadros = 0
                        gate.reset()
                        instruccion_principal = "SOLO UNA PERSONA A LA VEZ"
                        instruccion_secundaria = "Se detectaron múltiples personas en el encuadre"
                        color_guia = (0, 140, 255)  # Naranja
                    else:
                        face = faces[0]
                        cara_detectada = face
                        fx, fy, fw, fh = face[:4]
                        fcx, fcy = fx + fw / 2, fy + fh / 2

                        # Evaluación de postura, encuadre y nitidez
                        dx = abs(fcx - gcx)
                        dy = abs(fcy - gcy)

                        # Nitidez / anti-blur
                        cy1, cy2 = max(0, int(fy)), min(h, int(fy + fh))
                        cx1, cx2 = max(0, int(fx)), min(w, int(fx + fw))
                        crop_face = frame[cy1:cy2, cx1:cx2]
                        blur_val = cv2.Laplacian(cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() if crop_face.size > 0 else 0

                        # Pose frontal (evita escaneo con cabeza ladeada)
                        eye_x = sorted((float(face[4]), float(face[6])))
                        span = eye_x[1] - eye_x[0]
                        offset = (float(face[8]) - sum(eye_x) / 2) / span if span > 10 else 0

                        # 1. Distancia / Tamaño responsive
                        if min(fw, fh) < min_size or fw < gw * 0.45:
                            estabilidad_cuadros = 0
                            instruccion_principal = "ACÉRQUESE MÁS A LA CÁMARA"
                            instruccion_secundaria = "Su rostro está demasiado lejos para identificar"
                            color_guia = (0, 220, 255)
                        elif fw > gw * 1.15 or fh > gh * 1.15:
                            estabilidad_cuadros = 0
                            instruccion_principal = "ALÉJESE UN POCO DE LA CÁMARA"
                            instruccion_secundaria = "Su rostro está demasiado cerca del sensor"
                            color_guia = (0, 220, 255)
                        # 2. Centrado en el recuadro guía
                        elif dx > gw * 0.28 or dy > gh * 0.28 or fx < 0 or fy < 0 or fx + fw > w or fy + fh > h:
                            estabilidad_cuadros = 0
                            if fcx < gcx - gw * 0.28:
                                instruccion_principal = "MUEVA SU ROSTRO A LA DERECHA"
                            elif fcx > gcx + gw * 0.28:
                                instruccion_principal = "MUEVA SU ROSTRO A LA IZQUIERDA"
                            elif fcy < gcy - gh * 0.28:
                                instruccion_principal = "BAJE UN POCO LA CABEZA"
                            elif fcy > gcy + gh * 0.28:
                                instruccion_principal = "SUBA UN POCO LA CABEZA"
                            else:
                                instruccion_principal = "CENTRE SU ROSTRO EN EL RECUADRO"
                            instruccion_secundaria = "Alinee su rostro con la guía central"
                            color_guia = (0, 220, 255)
                        # 3. Nitidez y enfoque
                        elif blur_val < 32.0:
                            estabilidad_cuadros = 0
                            instruccion_principal = "MANTÉNGASE QUIETO (ENFOCANDO)"
                            instruccion_secundaria = "Imagen borrosa; evite movimientos bruscos"
                            color_guia = (0, 220, 255)
                        # 4. Mirar de frente
                        elif abs(offset) > 0.22:
                            estabilidad_cuadros = 0
                            instruccion_principal = "MIRE DE FRENTE A LA CÁMARA"
                            instruccion_secundaria = "Coloque su cabeza mirando al frente antes de escanear"
                            color_guia = (0, 220, 255)
                        else:
                            # Posición y enfoque óptimos: contar estabilidad
                            estabilidad_cuadros += 1
                            color_guia = (0, 255, 0)
                            if estabilidad_cuadros < REQUERIDOS_ESTABILIDAD:
                                instruccion_principal = f"ROSTRO ALINEADO. MANTÉNGASE QUIETO ({estabilidad_cuadros}/{REQUERIDOS_ESTABILIDAD})..."
                                instruccion_secundaria = "Enfoque óptimo alcanzado. Escaneando identidad..."
                            else:
                                rostro_listo = True

                        # ==============================================================
                        # 4. ESCANEO Y RESPUESTA DETERMINANTE
                        # ==============================================================
                        if rostro_listo:
                            if human_artifact is not None and not es_humano(human_artifact, frame):
                                instruccion_principal = "VERIFICANDO ROSTRO HUMANO..."
                                instruccion_secundaria = "Esperando detección humana válida"
                                color_guia = (0, 140, 255)
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

                                        # Evaluación PAD si está configurado
                                        live = True
                                        if pad.local is not None or pad.net is not None:
                                            live, _ = pad.evaluar(frame, face)

                                        if person in cfg["autorizados"] and live:
                                            # AUTORIZADO: BIENVENIDO EN VERDE
                                            nombre_formateado = person.replace('_', ' ').upper()
                                            resultado_bloqueo_hasta = now + 3.5
                                            resultado_tipo = "CONCEDIDO"
                                            resultado_mensaje = f"BIENVENIDO {nombre_formateado}"
                                            resultado_submensaje = "ACCESO CONCEDIDO - PUERTA ABIERTA"
                                            door.abrir()
                                            gate.next_allowed = now + cfg["rele"]["enfriamiento"]
                                            registrar_evento(f"Acceso concedido: {nombre_formateado}", person)
                                            gate.reset()
                                            estabilidad_cuadros = 0
                                        else:
                                            # NO AUTORIZADO: PUERTA CERRADA EN ROJO
                                            resultado_bloqueo_hasta = now + 3.0
                                            resultado_tipo = "DENEGADO"
                                            resultado_mensaje = "PUERTA CERRADA"
                                            resultado_submensaje = "ACCESO DENEGADO - PERSONA NO AUTORIZADA"
                                            registrar_evento("Puerta cerrada: persona no autorizada", person or "__desconocido__")
                                            gate.reset()
                                            estabilidad_cuadros = 0
                                except ValueError:
                                    pass

                # ==============================================================
                # 5. RENDERIZADO VISUAL RESPONSIVE (HUD)
                # ==============================================================
                if preview:
                    # Panel superior
                    superponer_panel(frame, 0, top_h, color=(12, 12, 12), alpha=0.75)
                    title_text = f"SISTEMA DE ACCESO FACIAL IA | Algoritmo: {algoritmo.upper()}"
                    sc_title = calcular_escala_texto(title_text, max_text_w, base_scale_title, grosor=1)
                    cv2.putText(frame, title_text, (margin_x, int(top_h * 0.38)),
                                cv2.FONT_HERSHEY_SIMPLEX, sc_title, (200, 200, 200), 1, cv2.LINE_AA)

                    sc_main = calcular_escala_texto(instruccion_principal, max_text_w, base_scale_main, grosor=2)
                    cv2.putText(frame, instruccion_principal, (margin_x, int(top_h * 0.84)),
                                cv2.FONT_HERSHEY_SIMPLEX, sc_main, color_guia, 2, cv2.LINE_AA)

                    # Recuadro central de encuadre
                    dibujar_esquinas(frame, gx1, gy1, gx2, gy2, color_guia, grosor=corner_thick, longitud=corner_len)

                    # Solo recuadro alrededor de la cara detectada
                    if cara_detectada is not None:
                        fx, fy, fw, fh = cara_detectada[:4].astype(int)
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color_guia, face_thick)

                    # Panel inferior
                    superponer_panel(frame, h - bot_h, h, color=(12, 12, 12), alpha=0.75)
                    sc_sub = calcular_escala_texto(instruccion_secundaria, max_text_w - 180, base_scale_sub, grosor=1)
                    cv2.putText(frame, instruccion_secundaria, (margin_x, h - int(bot_h * 0.38)),
                                cv2.FONT_HERSHEY_SIMPLEX, sc_sub, (255, 255, 255), 1, cv2.LINE_AA)

                    hint_text = "Q: Salir | F: Pantalla" if w >= 600 else "Q: Salir"
                    cv2.putText(frame, hint_text, (w - int(len(hint_text) * 11 * base_scale_title) - margin_x, h - int(bot_h * 0.38)),
                                cv2.FONT_HERSHEY_SIMPLEX, base_scale_title, (180, 180, 180), 1, cv2.LINE_AA)

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
