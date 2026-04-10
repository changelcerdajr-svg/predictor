# claude.md — Reglas de interacción

## Identidad del proyecto
- MLB betting pipeline (Python)
- Stack: LightGBM, pybaseball, Statcast, Action Network scraper
- Repo: changelcerdajr-svg/predictor

## Reglas de respuesta
- ULTRA conciso. Sin preamble.
- No explicar lo que ya es obvio en el código.
- No repetir código que ya existe salvo que sea el diff exacto.
- Priorizar: código accionable > explicación.
- Si algo no es crítico, omitirlo.
- Formato preferido: código en bloque → cambio mínimo → nada más.

## Formato para cambios
- Mostrar solo el fragmento afectado (no el archivo completo).
- Usar comentarios `# ANTES` / `# DESPUÉS` para diffs.
- Si el cambio es < 5 líneas, inline. Si es > 5 líneas, archivo completo.

## Contexto persistente
- `_fetch_odds` / `_stub_odds` → reemplazados por `pipeline/ingest/odds.py::fetch_odds()`
- Formato de odds: `{game_pk: {"odds_home": int, "odds_away": int}}`
- No usar The Odds API (eliminada). Fuente actual: Action Network.
- Statcast: pre-filtrado siempre por `game_date < as_of_date` (leakage check activo).
- Kelly: quarter-Kelly (0.25), min_edge=0.03, max_fraction=0.05.
- Drawdown halt: 10% desde peak.

## Anti-patrones (nunca hacer)
- No refactorizar lo que no se pidió.
- No agregar typing/docstrings donde no los hay.
- No sugerir tests salvo que se pidan.
- No dar alternativas salvo que haya un trade-off real.
- No usar emojis.

## Cuando el contexto no esté claro
- Pedir SOLO el archivo específico que falta.
- Una pregunta, no lista de preguntas.
