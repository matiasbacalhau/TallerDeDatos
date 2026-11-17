"""Corre todo el pipeline del TP3 en orden, con un solo comando.

Pensado para correr cada vez que llegan días nuevos a la carpeta de entrada: cada paso es
incremental o determinista, así que agregar datos no requiere tocar código. Se detiene en el
primer paso que falle (por ejemplo, si un control de causalidad no pasa).

  1. preparar_datos      CSV diarios -> data/pares/ (solo días nuevos o que cambiaron)
  2. calidad_dias        excluye días con faltantes anómalos, alarmas, data/calidad/REGISTRO_CALIDAD.md
  3. validacion          pliegues walk-forward sobre los días no excluidos
  4. baselines           B0-B3 en los pliegues -> data/resultados/
  5. features            variables por día -> data/features/
  6. verificar_features  causalidad y recálculo independiente
  7. producto            entrena el modelo elegido en config.yaml (producto.modelo) -> data/producto/

Todo lo que sale en pantalla queda también en salidas/salida_pipeline_DDMMYYYY.txt (una corrida
el mismo día reemplaza la anterior).

La parte experimental (elegir hiperparámetros y comparar modelos) no corre acá: son los scripts de src/modelos/.

Uso:  .venv/bin/python src/pipeline.py              # incremental
      .venv/bin/python src/pipeline.py --rehacer    # rehace data/pares/ de cero
"""
from datetime import date
from pathlib import Path
import subprocess
import sys
import time

SRC = Path(__file__).resolve().parent
PASOS = {"preparar_datos": "Preparar los datos", "calidad_dias": "Control de calidad por día",
         "validacion": "Esquema de validación", "baselines": "Baselines (correcciones simples a superar)",
         "features": "Variables para los modelos", "verificar_features": "Control independiente de las variables",
         "producto": "Modelo producto (el elegido en config.yaml)"}


def main():
    extra = ["--rehacer"] if "--rehacer" in sys.argv else []
    ruta = SRC.parent / "salidas" / f"salida_pipeline_{date.today():%d%m%Y}.txt"
    with open(ruta, "w", encoding="utf-8") as log:
        def decir(texto):                                # a pantalla y al archivo
            print(texto, flush=True)
            log.write(texto + "\n")
            log.flush()

        for i, (paso, titulo) in enumerate(PASOS.items(), 1):
            t = time.time()
            decir(f"\n{'=' * 100}\nPASO {i} de {len(PASOS)} · {titulo}   ({paso}.py)\n{'=' * 100}")
            args = [sys.executable, "-u", str(SRC / f"{paso}.py")] + (extra if paso == "preparar_datos" else [])
            proc = subprocess.Popen(args, cwd=SRC.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for linea in proc.stdout:
                decir(linea.rstrip("\n"))
            if proc.wait() != 0:
                decir(f"FALLÓ el paso {paso} (código {proc.returncode}); no se corren los siguientes.")
                sys.exit(1)
            decir(f"\n(paso {i} terminado en {time.time() - t:.0f} s)")
        decir(f"\n{'=' * 100}\nPipeline completo: los {len(PASOS)} pasos terminaron y todos los controles pasaron.\n"
              "Las líneas AVISO y ALARMA de arriba no detienen el pipeline, pero conviene leerlas.")
    print(f"Salida guardada en {ruta.relative_to(SRC.parent)}")


if __name__ == "__main__":
    main()
