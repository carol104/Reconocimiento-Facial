import csv
import json
from pathlib import Path


def comparar(reports, cfg):
    output = Path(cfg["artefactos"])
    (output / "comparacion.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    lines = []

    def emitir(line):
        print(line)
        lines.append(line)

    emitir("\nERRORES DE CLASIFICACIÓN (%)")
    emitir(f"{'Tarea':<12} {'Algoritmo':<17} {'Entrenamiento':>14} {'Validación':>12} {'Prueba':>12}")
    for report in reports:
        key = "error_clasificacion" if report.get("tarea") == "humanos" else "error_clasificacion_conocidos"
        values = [report["metricas"][s][key] for s in ("train", "val", "test")]
        emitir(f"{report.get('tarea', 'identidad'):<12} {report['algoritmo']:<17} " +
              " ".join(f"{v * 100:>12.2f}" for v in values))
        rows.append({"tarea": report.get("tarea", "identidad"), "algoritmo": report["algoritmo"],
            "error_train": values[0], "error_val": values[1], "error_test": values[2],
            "brecha_test_menos_train": values[2] - values[0],
            **{f"muestras_{s}": report["metricas"][s]["muestras"] for s in ("train", "val", "test")},
            **{f"error_acceso_con_calidad_{s}": report["metricas"][s].get("error_acceso_incluyendo_calidad")
               for s in ("train", "val", "test")},
            **{f"error_cascada_con_calidad_{s}": report["metricas"][s].get("error_cascada_incluyendo_calidad")
               for s in ("train", "val", "test")}})
    with (output / "comparacion.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cascades = [r for r in reports if "error_cascada_incluyendo_calidad" in r["metricas"]["test"]]
    if cascades:
        emitir("\nERROR HUMANO + IDENTIDAD, INCLUYENDO RECHAZOS POR CALIDAD (%)")
        emitir(f"{'Algoritmo':<17} {'Entrenamiento':>14} {'Validación':>12} {'Prueba':>12}")
        for report in cascades:
            values = [report["metricas"][s]["error_cascada_incluyendo_calidad"] for s in ("train", "val", "test")]
            emitir(f"{report['algoritmo']:<17} " + " ".join(f"{v * 100:>12.2f}" for v in values))
    (output / "resumen_errores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
