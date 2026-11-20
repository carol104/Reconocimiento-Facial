"""Módulo de captura de cámara, percepción computacional, reconocimiento multi-persona y control Arduino."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
def superponer_panel(frame, y1, y2, color=(16, 16, 16), alpha=0.75):
    """Dibuja una franja semitransparente adaptable para resaltar textos."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, max(0, y1)), (frame.shape[1], min(frame.shape[0], y2)), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def ejecutar(cfg, algoritmo, camera=0, hardware=False, preview=True):
    from clasificacion.algoritmos import NOMBRES
    from percepcion_computacional.vision import Vision, PAD
    from clasificacion.entrenamiento import cargar_artefacto, identificar
    from clasificacion.acceso import Puerta, verificar_hardware
    from clasificacion.arduino import ArduinoControlador

    LISTA_ALGORITMOS = ("svm", "knn", "ann", "cnn", "random_forest", "arbol_decision")
    algoritmo_actual = "svm" if algoritmo == "todos" else algoritmo

    artefactos_cache = {}
    humanos_cache = {}

    def obtener_artefacto(nombre):
        if nombre not in artefactos_cache:
            artefactos_cache[nombre] = cargar_artefacto(nombre, cfg)
        return artefactos_cache[nombre]

    def obtener_humano(nombre):
        if "fuentes" in cfg:
            if nombre not in humanos_cache:
                from percepcion_computacional.humanos import cargar_humanos
                humanos_cache[nombre] = cargar_humanos(nombre, cfg)
            return humanos_cache[nombre]
        return None

    def obtener_errores(art):
        m = art.get("reporte", {}).get("metricas", {})
        tr = m.get("train", {}).get("error_clasificacion_conocidos", 0.0)
        te = m.get("test", {}).get("error_clasificacion_conocidos", 0.0)
        tr_pct = f"{tr * 100:.2f}%" if tr is not None else "0.00%"
        te_pct = f"{te * 100:.2f}%" if te is not None else "0.00%"
        return tr_pct, te_pct

    artifact = obtener_artefacto(algoritmo_actual)
    human_artifact = obtener_humano(algoritmo_actual)
    train_err_pct, test_err_pct = obtener_errores(artifact)

    vision, pad = Vision(cfg), PAD(cfg["pad"])
    if hardware:
        verificar_hardware(cfg, artifact, pad.net is not None or pad.local is not None)

    # Inicializar controlador de Arduino
    arduino = ArduinoControlador(cfg)

    print("Modo GPIO (Hardware activado)" if hardware else "Modo simulación (sin GPIO)")
    print(f"\n[PERCEPCIÓN Y CLASIFICACIÓN ACTIVA] Algoritmo: {algoritmo_actual.upper()}")
    print("-> 1. Reconoce que son personas (filtro de objetos y animales).")
    print("-> 2. Clasifica si permitir o no (autorizados vs no autorizados).")
    print("-> 3. Percepción computacional: etiqueta con nombre y guarda capturas.")
    print("-> 4. Registra resultados y métricas de cada modelo continuamente.\n")

    capture = cv2.VideoCapture(camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Percepción Computacional y Control de Acceso IA"
    is_fullscreen = False

    # Control de salida mediante clic en la 'X'
    boton_x_pulsado = False
    btn_x1, btn_y1, btn_x2, btn_y2 = 0, 0, 0, 0

    def manejar_raton(event, x, y, flags, param):
        nonlocal boton_x_pulsado
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                boton_x_pulsado = True

    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, manejar_raton)

    # Carpetas y archivos de registro
    artefactos_dir = Path(cfg["artefactos"])
    artefactos_dir.mkdir(parents=True, exist_ok=True)
    capturas_dir = artefactos_dir / "capturas"
    capturas_dir.mkdir(parents=True, exist_ok=True)

    log_path = artefactos_dir / "accesos.jsonl"
    csv_path = artefactos_dir / "registro_accesos.csv"
    if not csv_path.exists():
        csv_path.write_text("Fecha_Hora,Algoritmo,Percepcion,Identidad,Decision_Acceso,Arduino_Senal,Error_Train,Error_Test,Captura\n", encoding="utf-8")

    last_log_state = None
    ultima_captura_tiempo = 0.0

    def cambiar_algoritmo(nuevo_nombre):
        nonlocal algoritmo_actual, artifact, human_artifact, train_err_pct, test_err_pct
        if nuevo_nombre in LISTA_ALGORITMOS:
            algoritmo_actual = nuevo_nombre
            artifact = obtener_artefacto(algoritmo_actual)
            human_artifact = obtener_humano(algoritmo_actual)
            train_err_pct, test_err_pct = obtener_errores(artifact)
            print(f"[CAMBIO EN VIVO] Algoritmo: {algoritmo_actual.upper()} (Train: {train_err_pct} | Test: {test_err_pct})")

    VERDE = (0, 255, 0)
    ROJO = (0, 0, 255)
    AMARILLO = (0, 220, 255)

    foco_encendido_hasta = 0.0
    DURACION_FOCO_SEGUNDOS = 5.0

    try:
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera}. Comprueba que esté conectada.")

        with Puerta(cfg["rele"], hardware) as door:
            while True:
                # Comprobar salida por clic en [X] o cierre de ventana del SO
                if boton_x_pulsado:
                    print("\n[SALIDA] Botón 'X' presionado. Cerrando programa...")
                    break

                if preview:
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n[SALIDA] Ventana cerrada por el usuario. Finalizando...")
                            break
                    except Exception:
                        pass

                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Se perdió la señal de la cámara; acceso detenido.")

                now = time.monotonic()
                h, w = frame.shape[:2]

                # Coordenadas responsive del botón [X] rojo
                btn_tam = max(28, int(h * 0.055))
                btn_x2 = w - 10
                btn_x1 = btn_x2 - btn_tam
                btn_y1 = 6
                btn_y2 = btn_y1 + btn_tam

                # ==============================================================
                # 1. PERCEPCIÓN COMPUTACIONAL: DETECCIÓN DE ROSTROS Y FILTRO HUMANO
                # ==============================================================
                vision.detector.setInputSize((w, h))
                _, faces = vision.detector.detect(frame)

                personas_evaluadas = []
                hay_autorizado_en_cuadro = False

                if faces is not None and len(faces) > 0:
                    from percepcion_computacional.humanos import es_humano

                    for face in faces:
                        fx, fy, fw, fh = face[:4].astype(int)
                        cx1, cy1 = max(0, fx), max(0, fy)
                        cx2, cy2 = min(w, fx + fw), min(h, fy + fh)
                        crop_rostro = frame[cy1:cy2, cx1:cx2]

                        if crop_rostro.size == 0:
                            continue

                        # 1.1 Filtro biológico: Clasificar si es humano vs objeto/animal
                        es_rostro_humano = True
                        if human_artifact is not None and not es_humano(human_artifact, crop_rostro):
                            es_rostro_humano = False
                            # Si es objeto o animal: NO pasa a la fase de reconocimiento
                            personas_evaluadas.append({
                                "box": (fx, fy, fw, fh),
                                "es_humano": False,
                                "autorizado": False,
                                "linea1": "[NO HUMANO: OBJETO/ANIMAL]",
                                "linea2": "NO PERMITIR",
                                "color": AMARILLO,
                                "nombre": None
                            })
                            continue

                        # ==============================================================
                        # 2. CLASIFICACIÓN DE ACCESO: RECONOCIMIENTO Y PERMITIR / NO
                        # ==============================================================
                        try:
                            vector, tensor = vision.representar_rostro(frame, face)
                            person, score, margin = identificar(artifact, vector, tensor)

                            # 2.1 Clasificar si permitir o no
                            es_aut = person in cfg["autorizados"]
                            if es_aut:
                                hay_autorizado_en_cuadro = True
                                nombre_fmt = person.replace('_', ' ').upper()
                                linea1 = f"[HUMANO] {nombre_fmt}"
                                linea2 = "PERMITIDO: PERSONA AUTORIZADA"
                                color = VERDE
                            else:
                                linea1 = "[HUMANO] DESCONOCIDO"
                                linea2 = "NO PERMITIDO: PERSONA NO AUTORIZADA"
                                color = ROJO

                            personas_evaluadas.append({
                                "box": (fx, fy, fw, fh),
                                "es_humano": True,
                                "autorizado": es_aut,
                                "linea1": linea1,
                                "linea2": linea2,
                                "color": color,
                                "nombre": person if es_aut else "DESCONOCIDO"
                            })
                        except Exception:
                            pass

                # ==============================================================
                # 3. CONTROL DE ARDUINO (TEMPORIZADOR DE 5 SEGUNDOS)
                # ==============================================================
                if hay_autorizado_en_cuadro:
                    foco_encendido_hasta = now + DURACION_FOCO_SEGUNDOS

                foco_activo = (now < foco_encendido_hasta)
                tiempo_restante_foco = max(0.0, foco_encendido_hasta - now)
                arduino.enviar_estado(foco_activo)

                if foco_activo and hardware:
                    door.abrir()

                # ==============================================================
                # 4. REGISTRO DE RESULTADOS DE CADA MODELO (JSONL, CSV Y CAPTURAS)
                # ==============================================================
                estado_actual = "ACTIVO" if foco_activo else "INACTIVO"
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

                # Guardar captura etiquetada en disco si hay personas detectadas (máx 1 cada 2 segundos)
                ruta_captura_rel = ""
                if len(personas_evaluadas) > 0 and (now - ultima_captura_tiempo >= 2.0):
                    nombre_primer_autorizado = next((p["nombre"] for p in personas_evaluadas if p["autorizado"]), "no_autorizado")
                    nombre_archivo_captura = f"{time.strftime('%Y%m%d_%H%M%S')}_{algoritmo_actual}_{nombre_primer_autorizado}.jpg"
                    ruta_guardado = capturas_dir / nombre_archivo_captura
                    cv2.imwrite(str(ruta_guardado), frame)
                    ruta_captura_rel = str(ruta_guardado)
                    ultima_captura_tiempo = now

                # Registrar en logs al cambiar estado o al detectar personas
                if estado_actual != last_log_state and len(personas_evaluadas) > 0:
                    for p in personas_evaluadas:
                        percepcion = "HUMANO" if p["es_humano"] else "OBJETO/ANIMAL"
                        identidad = p["nombre"] or "DESCONOCIDO"
                        decision = "PERMITIDO" if p["autorizado"] else "NO PERMITIDO"
                        senal = "1 (VERDADERO)" if foco_activo else "0 (FALSO)"

                        # Registro en JSONL
                        item_json = {
                            "fecha": timestamp_str,
                            "algoritmo": algoritmo_actual,
                            "percepcion": percepcion,
                            "identidad": identidad,
                            "decision": decision,
                            "arduino_senal": senal,
                            "error_train": train_err_pct,
                            "error_test": test_err_pct,
                            "captura": ruta_captura_rel
                        }
                        with log_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(item_json, ensure_ascii=False) + "\n")

                        # Registro en CSV para Excel
                        with csv_path.open("a", encoding="utf-8") as stream:
                            stream.write(f'"{timestamp_str}","{algoritmo_actual}","{percepcion}","{identidad}","{decision}","{senal}","{train_err_pct}","{test_err_pct}","{ruta_captura_rel}"\n')

                    last_log_state = estado_actual

                # ==============================================================
                # 5. RENDERIZADO VISUAL RESPONSIVE (ETIQUETADO CON NOMBRE)
                # ==============================================================
                if preview:
                    for p in personas_evaluadas:
                        fx, fy, fw, fh = p["box"]
                        color = p["color"]
                        linea1 = p["linea1"]
                        linea2 = p["linea2"]

                        # Cuadrito alrededor de la cara
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

                        # Escala adaptable de texto según ancho del rostro
                        escala_lbl = max(0.38, min(0.62, fw / 250.0))
                        (tw1, th1), _ = cv2.getTextSize(linea1, cv2.FONT_HERSHEY_SIMPLEX, escala_lbl, 2)
                        (tw2, th2), _ = cv2.getTextSize(linea2, cv2.FONT_HERSHEY_SIMPLEX, escala_lbl, 2)
                        ancho_caja = max(tw1, tw2) + 12
                        alto_caja = th1 + th2 + 14

                        # Posición de la etiqueta: arriba o abajo de la cara
                        lbl_y = fy - alto_caja - 4 if (fy - alto_caja - 4 > 10) else (fy + fh + 6)
                        lbl_x = max(4, fx)

                        # Fondo oscuro para legibilidad
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (lbl_x, lbl_y), (lbl_x + ancho_caja, lbl_y + alto_caja), (14, 14, 14), -1)
                        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

                        # Línea 1: [HUMANO] Nombre
                        cv2.putText(frame, linea1, (lbl_x + 6, lbl_y + th1 + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, escala_lbl, (255, 255, 255), 2, cv2.LINE_AA)
                        # Línea 2: PERMITIDO / NO PERMITIDO
                        cv2.putText(frame, linea2, (lbl_x + 6, lbl_y + th1 + th2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, escala_lbl, color, 2, cv2.LINE_AA)

                    # Barra superior: Algoritmo y Estado del Foco / Arduino
                    top_h = max(38, int(h * 0.070))
                    superponer_panel(frame, 0, top_h, color=(14, 14, 14), alpha=0.75)

                    sc_badge = max(0.40, min(0.55, w / 1600.0))
                    # Titulito del algoritmo en uso y sus errores
                    cv2.putText(frame, f"Modelo: {algoritmo_actual.upper()} | Train Err: {train_err_pct} | Test Err: {test_err_pct}",
                                (16, int(top_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, sc_badge, (220, 220, 220), 1, cv2.LINE_AA)

                    # Indicador de Arduino
                    if foco_activo:
                        txt_foco = f"ARDUINO: 1 (ACTIVO 5s) - {int(tiempo_restante_foco) + 1}s"
                        col_foco = VERDE
                    else:
                        txt_foco = "ARDUINO: 0 (INACTIVO)"
                        col_foco = (160, 160, 160)

                    (atw, _), _ = cv2.getTextSize(txt_foco, cv2.FONT_HERSHEY_SIMPLEX, sc_badge, 2)
                    cv2.putText(frame, txt_foco, (btn_x1 - atw - 24, int(top_h * 0.65)),
                                cv2.FONT_HERSHEY_SIMPLEX, sc_badge, col_foco, 2, cv2.LINE_AA)

                    # BOTÓN VISUAL [X] ROJO EN LA ESQUINA SUPERIOR DERECHA
                    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 210), -1)
                    margen_x = max(6, int(btn_tam * 0.22))
                    cv2.line(frame, (btn_x1 + margen_x, btn_y1 + margen_x),
                             (btn_x2 - margen_x, btn_y2 - margen_x), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (btn_x1 + margen_x, btn_y2 - margen_x),
                             (btn_x2 - margen_x, btn_y1 + margen_x), (255, 255, 255), 2, cv2.LINE_AA)

                    # Barra inferior discreta
                    bot_h = max(28, int(h * 0.05))
                    superponer_panel(frame, h - bot_h, h, color=(14, 14, 14), alpha=0.75)
                    hint_txt = "Clic en [X] o presiona 'Q' para Salir | F: Pantalla | TAB / 1-6: Cambiar Modelo" if w >= 680 else "[X] o 'Q': Salir"
                    cv2.putText(frame, hint_txt, (16, h - int(bot_h * 0.32)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    elif key in (ord("f"), ord("F")):
                        is_fullscreen = not is_fullscreen
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
                    elif key in (ord("\t"), ord("a"), ord("A")):
                        idx = (LISTA_ALGORITMOS.index(algoritmo_actual) + 1) % len(LISTA_ALGORITMOS)
                        cambiar_algoritmo(LISTA_ALGORITMOS[idx])
                    elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                        idx = int(chr(key)) - 1
                        if 0 <= idx < len(LISTA_ALGORITMOS):
                            cambiar_algoritmo(LISTA_ALGORITMOS[idx])
    finally:
        arduino.cerrar()
        capture.release()
        if preview:
            cv2.destroyAllWindows()
