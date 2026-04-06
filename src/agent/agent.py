"""
src/agent/agent.py — Agente conversacional con sistema de feedback
Uso: python src/agent/agent.py  (desde la raíz del proyecto)

Requiere:
  - ANTHROPIC_API_KEY en .env
  - outputs/predictions/predicciones.parquet generado por main.py
"""

import os
import re
import sys
import json
import datetime
from pathlib import Path
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass

import anthropic
import pandas as pd
import duckdb
import yaml

# ─── CONFIG ────────────────────────────────────────────────────────────────────

def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG         = _load_config()
MODEL       = CFG["agent"]["model"]
MAX_TOKENS  = CFG["agent"]["max_tokens"]
PRED_FILE   = CFG["paths"]["predictions"]
FEEDBACK_FILE = CFG["agent"].get("feedback_file", "outputs/agent_feedback.jsonl")
THRESHOLD   = CFG["agent"].get("high_propensity_threshold", 0.6)
FEACTURE_IMPORTANCE_FILE = CFG["paths"].get("feactures_importance", "outputs/feature_importance/feature_importance.csv")

# ─── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
Eres un analista senior de datos especializado en comportamiento de compra retail para Ferreycorp.

DATASET: 58,693 visitas de clientes con variables de precio, promociones, perfil demográfico
y comportamiento histórico. Target: incidencia_compra (0/1).

TABLA DE PREDICCIONES:
  id, cluster, prob_compra (0-1), prediccion (0/1), score_decil (1=mayor propensión),
  edad, ingreso_anual, genero, top_feature_1/2/3

INSTRUCCIONES DE COMPORTAMIENTO:
- Responde siempre en español
- Sé directo y conciso: máximo 4-5 líneas de interpretación
- Usa números con 2 decimales
- Cuando muestres datos → interprétalos con visión de negocio, no solo los repitas
- Da una recomendación accionable al final de cada respuesta sobre datos
- Alta propensión = prob_compra > {THRESHOLD}
- Las variables más importantes para la compra son top_feature_1/2/3 (según SHAP) o revisar {FEACTURE_IMPORTANCE_FILE}
- Si la pregunta es ambigua, asume la interpretación más útil para el negocio
- NO expliques lo que vas a hacer, hazlo directamente
"""

SQL_PROMPT = """
Convierte la pregunta a SQL válido para DuckDB.
Tabla: predicciones
Columnas: id, cluster, prob_compra, prediccion, score_decil, edad, ingreso_anual, genero,
          top_feature_1, top_feature_2, top_feature_3

REGLAS:
- SOLO SQL, sin explicaciones ni backticks
- prob_compra es decimal (0.7, no 70%)
- Alta propensión = prob_compra > {threshold}
- LIMIT 20 para listados individuales
- GROUP BY + AVG/COUNT/SUM para resúmenes

Pregunta: {{pregunta}}
""".format(threshold=THRESHOLD)

# ─── INIT ──────────────────────────────────────────────────────────────────────

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ No se encontró ANTHROPIC_API_KEY en .env")
    sys.exit(1)

client    = anthropic.Anthropic(api_key=api_key)
historial = []
feature_importance_df = None

if Path(FEACTURE_IMPORTANCE_FILE).exists():
    feature_importance_df = pd.read_csv(FEACTURE_IMPORTANCE_FILE)
    print(f"✅ Feature importance cargado: {len(feature_importance_df)} variables")
else:
    print(f"⚠️ No se encontró {FEACTURE_IMPORTANCE_FILE}")

pred_df = None
if Path(PRED_FILE).exists():
    pred_df = pd.read_parquet(PRED_FILE)
    print(f"✅ Predicciones cargadas: {len(pred_df):,} registros")
else:
    print(f"⚠️  No se encontró {PRED_FILE} — ejecuta primero: python main.py")

Path(FEEDBACK_FILE).parent.mkdir(parents=True, exist_ok=True)

# ─── KEYWORDS ──────────────────────────────────────────────────────────────────

DATA_KEYWORDS = [
    "clientes", "segmento", "cluster", "propensión", "probabilidad", "predicción",
    "cuántos", "quiénes", "listar", "mostrar", "tabla", "filtrar", "top", "decil",
    "mayor", "menor", "comprar", "dame", "resumen", "promedio", "cuántas", "edad",
    "ingreso", "género", "muéstrame", "analiza", "compara", "distribución",
]


def es_consulta_datos(texto: str) -> bool:
    return pred_df is not None and any(k in texto.lower() for k in DATA_KEYWORDS)


def es_consulta_features(texto: str) -> bool:
    keywords = [
        "variable", "importancia", "influyen", "influye",
        "importante", "shap", "drivers", "factores"
    ]
    return feature_importance_df is not None and any(k in texto.lower() for k in keywords)



# ─── SQL ───────────────────────────────────────────────────────────────────────

def _generar_sql(pregunta: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": SQL_PROMPT.format(pregunta=pregunta)}],
    )
    return re.sub(r"```sql|```", "", resp.content[0].text).strip()


def ejecutar_sql(pregunta: str):
    sql = _generar_sql(pregunta)
    try:
        con = duckdb.connect()
        con.register("predicciones", pred_df)
        result = con.execute(sql).df()
        return sql, result, None
    except Exception as e:
        # Autocorrección
        fix = client.messages.create(
            model=MODEL, max_tokens=512,
            messages=[{"role": "user", "content":
                f"SQL con error '{e}':\n{sql}\nCorrígelo. Solo SQL."}],
        )
        sql2 = re.sub(r"```sql|```", "", fix.content[0].text).strip()
        try:
            con2 = duckdb.connect()
            con2.register("predicciones", pred_df)
            result = con2.execute(sql2).df()
            return sql2, result, None
        except Exception as e2:
            return sql, None, str(e2)


# ─── FEEDBACK ──────────────────────────────────────────────────────────────────

def registrar_feedback(pregunta: str, respuesta: str, rating: int, comentario: str = ""):
    """
    Guarda feedback del equipo en JSONL para análisis posterior.
    rating: 1 (malo) a 5 (excelente)
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pregunta": pregunta,
        "respuesta_preview": respuesta[:300],
        "rating": rating,
        "comentario": comentario,
        "model": MODEL,
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  📝 Feedback registrado (rating={rating})")


def solicitar_feedback(pregunta: str, respuesta: str):
    """Solicita feedback opcional al usuario después de cada respuesta."""
    try:
        raw = input("\n  [Feedback] ¿Útil esta respuesta? (1-5 / Enter para omitir): ").strip()
        if not raw:
            return
        rating = int(raw)
        if not 1 <= rating <= 5:
            return
        comentario = input("  [Feedback] Comentario opcional (Enter para omitir): ").strip()
        registrar_feedback(pregunta, respuesta, rating, comentario)
    except (ValueError, EOFError, KeyboardInterrupt):
        pass


def ver_estadisticas_feedback():
    """Muestra resumen del feedback acumulado."""
    if not Path(FEEDBACK_FILE).exists():
        print("  Sin feedback registrado aún.")
        return
    entries = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    if not entries:
        print("  Sin feedback registrado aún.")
        return
    ratings = [e["rating"] for e in entries]
    print(f"\n  📊 ESTADÍSTICAS DE FEEDBACK")
    print(f"     Total respuestas evaluadas: {len(ratings)}")
    print(f"     Rating promedio: {sum(ratings)/len(ratings):.2f}/5")
    print(f"     Distribución: " + " | ".join(f"{r}★:{ratings.count(r)}" for r in range(1, 6)))
    poor = [e for e in entries if e["rating"] <= 2 and e["comentario"]]
    if poor:
        print(f"\n  ⚠️  Últimas respuestas con rating ≤ 2:")
        for e in poor[-3:]:
            print(f"     [{e['timestamp'][:10]}] {e['pregunta'][:60]}... → '{e['comentario']}'")


# ─── RESPUESTA ─────────────────────────────────────────────────────────────────

def responder(user_input: str) -> str:
    full_response = ""

    if es_consulta_features(user_input):
        top_n = 10

        df_top = feature_importance_df.sort_values("shap", ascending=False).head(top_n)

        print("\n  📊 Top variables más importantes:\n")
        print(df_top.to_string(index=False))

        resumen = df_top.head(5).to_string(index=False)

        prompt = (
            f"Pregunta del equipo: '{user_input}'\n\n"
            f"Top variables según SHAP:\n{resumen}\n\n"
            f"Como analista senior, explica en 3-4 líneas qué variables influyen más "
            f"y qué significa esto para el negocio. Termina con UNA recomendación accionable."
        )

        historial.append({"role": "user", "content": prompt})

    if es_consulta_datos(user_input):
        sql, result, error = ejecutar_sql(user_input)
        print(f"\n  📋 SQL: {sql}\n")

        if error:
            print(f"  ❌ Error SQL: {error}")
            historial.append({"role": "user", "content": user_input})
        else:
            if result is not None and not result.empty:
                print(result.head(20).to_string(index=False))
                if len(result) > 20:
                    print(f"  ... ({len(result) - 20} filas adicionales)")
            else:
                print("  (sin resultados)")

            n = len(result) if result is not None else 0
            muestra = result.head(5).to_string() if result is not None and not result.empty else "vacío"
            prompt = (
                f"Pregunta del equipo: '{user_input}'\n"
                f"La consulta devolvió {n} registros. Muestra:\n{muestra}\n\n"
                f"Como analista senior, interpreta el resultado en 3-4 líneas con foco en negocio "
                f"y termina con UNA recomendación accionable concreta."
            )
            historial.append({"role": "user", "content": prompt})
    else:
        historial.append({"role": "user", "content": user_input})

    # Streaming
    print("\n💬 ", end="", flush=True)
    with client.messages.stream(
        model=MODEL, max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT, messages=historial,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text
    print()

    historial.append({"role": "assistant", "content": full_response})
    return full_response


# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║      🛒  AGENTE DE PROPENSIÓN DE COMPRA — Ferreycorp             ║
║                                                                  ║
║  Ejemplos:                                                       ║
║  • ¿Cuántos clientes tienen prob > 70%?                          ║
║  • Dame el top 10 del cluster 0 por propensión                   ║
║  • Compara propensión promedio por género y cluster              ║
║  • ¿Qué variables influyen más en la decisión de compra?         ║
║                                                                  ║
║  Comandos: /limpiar  /feedback  /salir                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

PEDIR_FEEDBACK = True   # Cambiar a False para deshabilitar feedback interactivo


def main():
    print(BANNER)

    while True:
        try:
            user_input = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego.")
            break

        if not user_input:
            continue

        if user_input == "/salir":
            print("👋 Hasta luego.")
            break

        if user_input == "/limpiar":
            historial.clear()
            print("🗑️  Conversación reiniciada.\n")
            continue

        if user_input == "/feedback":
            ver_estadisticas_feedback()
            print()
            continue

        respuesta = responder(user_input)

        if PEDIR_FEEDBACK:
            solicitar_feedback(user_input, respuesta)

        print("\n" + "─" * 66 + "\n")


if __name__ == "__main__":
    main()
