"""CLI común: preparar, entrenar, comparar y ejecutar en Raspberry Pi."""
import argparse
import json
from pathlib import Path
import signal
from urllib.error import URLError
from algoritmos import NOMBRES
from sistema.config import cargar


def parser():
    root = argparse.ArgumentParser(description="Acceso facial de cuatro personas")
    root.add_argument("--config", default="config.json")
    sub = root.add_subparsers(dest="comando", required=True)
    sub.add_parser("preparar", help="Separar las fuentes por sesiones y crear manifiesto")
    sub.add_parser("descargar-modelos", help="Descargar YuNet y SFace oficiales")
    pad = sub.add_parser("entrenar-pad", help="Entrenar CNN antispoofing con reales y ataques")
    pad.add_argument("--datos", default="dataset_pad")
    pad.add_argument("--epocas", type=int, default=30)
    train = sub.add_parser("entrenar", help="Entrenar, calibrar e imprimir errores")
    train.add_argument("--algoritmo", choices=(*NOMBRES, "todos"), default="todos")
    train.add_argument("--tarea", choices=("todos", "humanos", "identidad"), default="todos")
    run = sub.add_parser("ejecutar", help="Reconocer desde cámara; simulación por defecto")
    run.add_argument("--algoritmo", choices=(*NOMBRES, "todos"), default="todos")
    run.add_argument("--camara", type=int, default=0)
    run.add_argument("--hardware", action="store_true", help="Activar relé si pasan las verificaciones")
    run.add_argument("--sin-ventana", action="store_true")
    run.add_argument("--no-entrenar", action="store_true", help="Omitir reentrenamiento previo a la cámara")
    return root


def main():
    args = parser().parse_args()
    cfg = cargar(args.config)
    if args.comando == "preparar":
        if "fuentes" in cfg:
            from sistema.preparacion import preparar_fuentes
            preparar_fuentes(cfg)
            return
        for split in ("train", "val", "test"):
            for person in cfg["autorizados"]:
                (Path(cfg["dataset"]) / split / person / f"sesion_{split}").mkdir(parents=True, exist_ok=True)
            if split != "train":
                (Path(cfg["dataset"]) / split / "desconocidos" / f"extra_{split}" / f"sesion_{split}").mkdir(parents=True, exist_ok=True)
        print("Estructura creada. Añade imágenes de sesiones independientes; consulta README.md.")
    elif args.comando == "descargar-modelos":
        from sistema.modelos import descargar
        descargar(cfg)
    elif args.comando == "entrenar-pad":
        from sistema.vision import Vision
        from sistema.entrenar_pad import entrenar_pad
        if args.epocas < 1:
            raise ValueError("El número de épocas debe ser positivo.")
        entrenar_pad(cfg, args.datos, Vision(cfg), args.epocas)
    elif args.comando == "entrenar":
        from sistema.vision import Vision
        from sistema.datos import cargar_dataset
        from sistema.entrenamiento import entrenar
        names = NOMBRES if args.algoritmo == "todos" else (args.algoritmo,)
        if "cnn" in names:
            import importlib.util
            if importlib.util.find_spec("torch") is None:
                raise ValueError("Para CNN instala requirements-cnn.txt antes de entrenar.")
        reports = []
        if "fuentes" in cfg:
            from sistema.preparacion import cargar_fuentes
            from sistema.humanos import entrenar_humanos
            data = cargar_fuentes(cfg, Vision(cfg))
            for name in names:
                human_artifact = None
                if args.tarea in ("todos", "humanos"):
                    report, human_artifact = entrenar_humanos(name, data["humanos"], cfg)
                    reports.append(report)
                if args.tarea in ("todos", "identidad"):
                    reports.append(entrenar(name, data["identidad"], cfg, humanos=human_artifact))
        else:
            if args.tarea == "humanos":
                raise ValueError("Configura fuentes.humanos para entrenar humano/objeto.")
            data = cargar_dataset(cfg, Vision(cfg))
            reports = [entrenar(name, data, cfg) for name in names]
        from sistema.reportes import comparar
        comparar(reports, cfg)
        print(f"Informes guardados en {cfg['artefactos']}")
    else:
        if not args.no_entrenar:
            print("\n" + "=" * 62)
            print(f" [PASO 1/2] ENTRENANDO Y TESTEANDO EL MODELO: {args.algoritmo.upper()}")
            print("=" * 62)
            from sistema.vision import Vision
            from sistema.entrenamiento import entrenar
            names = NOMBRES if args.algoritmo == "todos" else (args.algoritmo,)
            if "cnn" in names:
                import importlib.util
                if importlib.util.find_spec("torch") is None:
                    raise ValueError("Para CNN instala requirements-cnn.txt antes de entrenar.")
            vision = Vision(cfg)
            if "fuentes" in cfg:
                from sistema.preparacion import cargar_fuentes
                from sistema.humanos import cargar_humanos, entrenar_humanos
                data = cargar_fuentes(cfg, vision)
                for name in names:
                    human_path = Path(cfg["artefactos"]) / f"humanos_{name}.joblib"
                    if human_path.exists():
                        human_art = cargar_humanos(name, cfg)
                    else:
                        _, human_art = entrenar_humanos(name, data["humanos"], cfg)
                    entrenar(name, data["identidad"], cfg, humanos=human_art)
            else:
                from sistema.datos import cargar_dataset
                data = cargar_dataset(cfg, vision)
                for name in names:
                    entrenar(name, data, cfg)
            print("=" * 62)
            print(" Entrenamiento y testeo completados exitosamente.")
            print("=" * 62)

        print("\n" + "=" * 62)
        print(f" [PASO 2/2] ABRIENDO CÁMARA PARA DETECCIÓN EN VIVO")
        print("=" * 62 + "\n")
        from sistema.camara import ejecutar
        ejecutar(cfg, args.algoritmo, args.camara, args.hardware, not args.sin_ventana)


def detener(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, detener)
    try:
        main()
    except KeyboardInterrupt:
        print("Sistema detenido; relé desactivado.")
    except (ValueError, OSError, RuntimeError, URLError) as exc:
        raise SystemExit(f"Error: {exc}")
