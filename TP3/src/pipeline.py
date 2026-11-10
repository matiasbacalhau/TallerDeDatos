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

Uso:  .venv/bin/python src/pipeline.py              # incremental
      .venv/bin/python src/pipeline.py --rehacer    # rehace data/pares/ de cero
"""
from pathlib import Path
import subprocess
import sys
import time

SRC = Path(__file__).resolve().parent
PASOS = ["preparar_datos", "calidad_dias", "validacion", "baselines", "features", "verificar_features"]


def main():
    extra = ["--rehacer"] if "--rehacer" in sys.argv else []
    for paso in PASOS:
        t = time.time()
        print(f"\n===== {paso} =====", flush=True)
        args = [sys.executable, str(SRC / f"{paso}.py")] + (extra if paso == "preparar_datos" else [])
        r = subprocess.run(args, cwd=SRC.parent)
        if r.returncode != 0:
            sys.exit(f"FALLÓ el paso {paso} (código {r.returncode}); no se corren los siguientes.")
        print(f"----- {paso}: {time.time() - t:.0f} s", flush=True)
    print("\nPipeline completo. Revisar data/calidad/REGISTRO_CALIDAD.md por avisos y alarmas.")


if __name__ == "__main__":
    main()
