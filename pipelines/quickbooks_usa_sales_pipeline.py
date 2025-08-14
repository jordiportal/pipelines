"""
title: QuickBooks USA Sales Pipeline
author: assistant
date: 2025-08-14
version: 1.0
license: MIT
description: Pipeline que consulta ventas de QuickBooks USA vía un servicio OpenAPI; obtiene el esquema, genera SQL con un LLM local (Ollama) y ejecuta la consulta.
requirements: requests
"""

from typing import List, Union, Generator, Iterator, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

import requests
import time
import json as jsonlib
import re


class Pipeline:
    class Valves(BaseModel):
        # === Configuración del servicio OpenAPI que expone QuickBooks DB ===
        OPENAPI_BASE_URL: str = Field(
            default="http://localhost:8000",
            description="Base URL del servicio OpenAPI que provee acceso a la BD agregada de QuickBooks",
        )
        SCHEMA_ENDPOINT_PATH: str = Field(
            default="/schema",
            description="Ruta del endpoint que devuelve el esquema completo de la BD",
        )
        REFRESH_ENDPOINT_PATH: str = Field(
            default="/refresh",
            description="Ruta del endpoint para forzar actualización de datos desde QuickBooks Online",
        )
        SQL_ENDPOINT_PATH: str = Field(
            default="/sql",
            description="Ruta del endpoint que ejecuta una consulta SQL (SELECT) y devuelve resultados",
        )
        OPENAPI_AUTH_HEADER: str = Field(
            default="Authorization",
            description="Nombre del header para autenticación contra el servicio OpenAPI",
        )
        OPENAPI_AUTH_TOKEN_PREFIX: str = Field(
            default="Bearer ",
            description="Prefijo del token en el header de autenticación (por ejemplo 'Bearer ')",
        )
        OPENAPI_AUTH_TOKEN: str = Field(
            default="",
            description="Token/API Key para el servicio OpenAPI",
        )

        # === Configuración de Ollama (LLM local para generación de SQL) ===
        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434",
            description="Base URL del servidor Ollama local",
        )
        OLLAMA_MODEL: str = Field(
            default="llama3.1:8b",
            description="Modelo de Ollama para generar consultas SQL seguras",
        )
        OLLAMA_TIMEOUT: int = Field(
            default=60,
            description="Timeout en segundos para llamadas al servidor Ollama",
        )

        # === Parámetros funcionales ===
        DEFAULT_COUNTRY: str = Field(
            default="USA",
            description="País por defecto para filtrar ventas (por ejemplo 'USA' o 'US')",
        )
        AUTO_REFRESH_BEFORE_QUERY: bool = Field(
            default=False,
            description="Si está en true, ejecuta un refresh de datos antes de consultar",
        )
        MAX_ROWS: int = Field(
            default=1000,
            description="Límite máximo de filas a devolver por consulta (se aplicará si el SQL no lo especifica)",
        )
        SCHEMA_CACHE_TTL: int = Field(
            default=900,
            description="TTL en segundos para cachear el esquema en memoria",
        )
        ALLOWED_TABLES: List[str] = Field(
            default_factory=list,
            description="Lista blanca de tablas permitidas. Vacío = todas",
        )
        ENFORCE_COUNTRY_FILTER: bool = Field(
            default=False,
            description="Si es true, intenta aplicar un filtro por país USA al SQL generado aunque el LLM no lo incluya",
        )
        # === Tool-calling y chat general ===
        TOOLS_ENABLED: bool = Field(
            default=True,
            description="Habilita que el LLM decida invocar herramientas (execute_sql, refresh_data, get_schema) devolviendo JSON",
        )
        INCLUDE_SCHEMA_IN_PROMPT: bool = Field(
            default=True,
            description="Incluye el esquema (truncado) en el prompt para mejor razonamiento",
        )
        SCHEMA_MAX_CHARS: int = Field(
            default=20000,
            description="Máximo de caracteres del esquema a incluir en el prompt",
        )
        ALLOW_GENERAL_CHAT: bool = Field(
            default=True,
            description="Permite que el pipeline responda conversaciones generales sin consultar la BD",
        )
        REQUIRE_EXPLICIT_CHART_REQUEST: bool = Field(
            default=True,
            description="Incluye código de gráficos solo si el usuario lo solicita explícitamente",
        )
        DECIMAL_PLACES: int = Field(
            default=1,
            description="Número de decimales a mostrar en valores numéricos",
        )
        SALES_KEYWORDS: List[str] = Field(
            default_factory=lambda: [
                "venta",
                "ventas",
                "sales",
                "ingresos",
                "factura",
                "invoices",
                "quickbooks",
                "qbo",
            ],
            description="Palabras clave para detectar intención de ventas QuickBooks USA",
        )

    def __init__(self):
        self.id = "quickbooks_usa_sales"
        self.name = "QuickBooks USA Sales"
        self.valves = self.Valves()

        # Cache de esquema en memoria
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._schema_cached_at: float = 0.0

    # =====================
    # Hooks de ciclo de vida
    # =====================
    async def on_startup(self):
        print(f"on_startup: {__name__}")
        # Intentar precargar el esquema (no bloqueante si falla)
        try:
            _ = self._get_schema(force=False)
        except Exception as e:
            print(f"⚠️ No se pudo precargar esquema: {e}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        # Invalidar cache si cambia configuración
        self._schema_cache = None
        self._schema_cached_at = 0.0

    # =====================
    # Utilidades HTTP
    # =====================
    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.valves.OPENAPI_AUTH_TOKEN:
            headers[self.valves.OPENAPI_AUTH_HEADER] = (
                f"{self.valves.OPENAPI_AUTH_TOKEN_PREFIX}{self.valves.OPENAPI_AUTH_TOKEN}"
            )
        return headers

    def _compose_url(self, path: str) -> str:
        base = self.valves.OPENAPI_BASE_URL.rstrip("/")
        p = path if path.startswith("/") else f"/{path}"
        return f"{base}{p}"

    # =====================
    # Esquema y refresh
    # =====================
    def _get_schema(self, force: bool = False) -> Dict[str, Any]:
        # Cache simple con TTL
        if (
            not force
            and self._schema_cache is not None
            and (time.time() - self._schema_cached_at) < self.valves.SCHEMA_CACHE_TTL
        ):
            return self._schema_cache

        url = self._compose_url(self.valves.SCHEMA_ENDPOINT_PATH)
        r = requests.get(url, headers=self._build_headers(), timeout=30)
        r.raise_for_status()
        schema = r.json()

        self._schema_cache = schema
        self._schema_cached_at = time.time()
        return schema

    def _refresh_data(self) -> Dict[str, Any]:
        url = self._compose_url(self.valves.REFRESH_ENDPOINT_PATH)
        r = requests.post(url, headers=self._build_headers(), timeout=60)
        r.raise_for_status()
        return r.json() if r.text else {"status": "ok"}

    # =====================
    # Generación y validación de SQL
    # =====================
    def _generate_sql_with_ollama(self, user_message: str, schema: Dict[str, Any]) -> str:
        system_prompt = (
            "Eres un asistente que genera exclusivamente consultas SQL de solo lectura (SELECT). "
            "Usa únicamente tablas y columnas que existan en el esquema proporcionado. "
            "No modifiques datos. No uses DDL ni DML. Incluye un LIMIT apropiado. "
            "Si el usuario pide ventas de USA, incluye un filtro de país solo si existe una columna correspondiente (por ejemplo, country, country_code, market) en el esquema. "
            "Devuelve SOLO el SQL sin explicaciones ni formato adicional."
        )

        schema_compact = jsonlib.dumps(schema)[: self.valves.SCHEMA_MAX_CHARS]  # limitar tamaño del prompt

        user_prompt = (
            f"Esquema JSON (parcial):\n{schema_compact}\n\n"
            f"Consulta del usuario: {user_message}\n\n"
            f"Requisitos: \n"
            f"- Solo SELECT\n- Si y solo si el esquema lo permite, incluir filtro por país = '{self.valves.DEFAULT_COUNTRY}'\n"
            f"- Incluir LIMIT {self.valves.MAX_ROWS} si el usuario no especifica límite\n"
        )

        try:
            r = requests.post(
                f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": self.valves.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "temperature": 0.1,
                },
                timeout=self.valves.OLLAMA_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Extraer SQL limpio (si viene con backticks o bloques)
            sql = self._extract_sql_from_text(content)
            self._validate_sql(sql)
            if self.valves.ENFORCE_COUNTRY_FILTER:
                sql = self._ensure_country_filter(sql)
            sql = self._ensure_limit(sql)
            return sql
        except Exception as e:
            raise Exception(f"Error generando SQL con Ollama: {e}")

    def _extract_sql_from_text(self, text: str) -> str:
        # El modelo puede devolver ```sql ... ``` o solo texto. Intentar limpiar.
        code_block = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if code_block:
            return code_block.group(1).strip().rstrip(";")
        return text.strip().rstrip(";")

    def _validate_sql(self, sql: str) -> None:
        # Asegurar que es SELECT y que no contiene instrucciones peligrosas
        if not re.match(r"^\s*select\b", sql, flags=re.IGNORECASE):
            raise ValueError("Solo se permiten consultas SELECT")
        forbidden = [";", "--", " drop ", " delete ", " update ", " insert ", " alter ", " create "]
        lowered = f" {sql.lower()} "
        for f in forbidden:
            if f in lowered:
                raise ValueError("El SQL contiene tokens no permitidos")

        # Validación de tablas si hay lista blanca
        if self.valves.ALLOWED_TABLES:
            # Extraer posibles referencias a tablas simples: FROM <tabla> o JOIN <tabla>
            tables = set(re.findall(r"\bfrom\s+([a-zA-Z0-9_\.]+)\b|\bjoin\s+([a-zA-Z0-9_\.]+)\b", sql, flags=re.IGNORECASE))
            flat_tables = {t for pair in tables for t in pair if t}
            for t in flat_tables:
                base = t.split(".")[-1]
                if base not in self.valves.ALLOWED_TABLES:
                    raise ValueError(f"Tabla no permitida en SQL: {t}")

    def _ensure_country_filter(self, sql: str) -> str:
        # Si ya hay un filtro de país, no hacer nada. Heurística simple
        if re.search(r"\b(country|country_code|market)\b\s*=\s*'?(us|usa)'?", sql, flags=re.IGNORECASE):
            return sql
        # Si hay un WHERE, agregar AND; si no, agregar WHERE
        if re.search(r"\bwhere\b", sql, flags=re.IGNORECASE):
            return re.sub(
                r"\bwhere\b",
                f"WHERE (country = '{self.valves.DEFAULT_COUNTRY}' OR country_code IN ('US','USA') OR market IN ('US','USA')) AND",
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            # Insertar antes de GROUP BY/ORDER BY/LIMIT si existen
            m = re.search(r"\b(group\s+by|order\s+by|limit)\b", sql, flags=re.IGNORECASE)
            condition = f" WHERE (country = '{self.valves.DEFAULT_COUNTRY}' OR country_code IN ('US','USA') OR market IN ('US','USA')) "
            if m:
                idx = m.start()
                return sql[:idx] + condition + sql[idx:]
            return sql + condition

    def _ensure_limit(self, sql: str) -> str:
        if re.search(r"\blimit\b\s+\d+", sql, flags=re.IGNORECASE):
            return sql
        return sql.rstrip() + f" LIMIT {self.valves.MAX_ROWS}"

    # =====================
    # Tool-calling (ligero)
    # =====================
    def _ollama_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Dict[str, Any]:
        r = requests.post(
            f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": self.valves.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
            },
            timeout=self.valves.OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def _decide_tool(self, user_message: str, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pide al LLM que decida si debe llamar una herramienta.
        Formato esperado (JSON plano en el contenido):
        {"tool":"execute_sql","sql":"SELECT ..."}
        o {"tool":"refresh_data"}
        o {"tool":"get_schema"}
        o {"tool":"none","response":"..."}
        """

        schema_snippet = jsonlib.dumps(schema)[: self.valves.SCHEMA_MAX_CHARS] if (schema and self.valves.INCLUDE_SCHEMA_IN_PROMPT) else ""
        system = (
            "Eres un asistente con acceso a herramientas. Si la pregunta requiere datos de la BD, "
            "devuelve un JSON con un tool call. Si no, responde directamente. Importante: devuelve SOLO JSON cuando llames una tool."
        )
        tools_desc = (
            "Herramientas disponibles:\n"
            "- execute_sql(sql: string): ejecuta una consulta SELECT en la BD. Debe ser segura y acorde al esquema.\n"
            "- refresh_data(): refresca datos desde QuickBooks Online.\n"
            "- get_schema(): devuelve el esquema actual (útil para razonar).\n"
            "Reglas: Solo SELECT. Incluye LIMIT si procede. Incluye filtro de país solo si el esquema lo soporta."
        )

        content_user = (
            (f"Esquema (parcial):\n{schema_snippet}\n\n" if schema_snippet else "")
            + f"Pregunta del usuario: {user_message}\n\n"
            + "Responde con UNO de los siguientes JSON (sin texto adicional):\n"
            + "- {\"tool\":\"execute_sql\",\"sql\":\"SELECT ...\"}\n"
            + "- {\"tool\":\"refresh_data\"}\n"
            + "- {\"tool\":\"get_schema\"}\n"
            + "- {\"tool\":\"none\",\"response\":\"respuesta directa al usuario\"}"
        )

        data = self._ollama_chat(
            [
                {"role": "system", "content": system},
                {"role": "system", "content": tools_desc},
                {"role": "user", "content": content_user},
            ]
        )
        raw = data["choices"][0]["message"]["content"].strip()
        # Intentar parsear JSON puro; si viene con texto extra, buscar primer bloque JSON
        parsed: Dict[str, Any]
        try:
            parsed = jsonlib.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                # No es tool-call
                return {"tool": "none", "response": raw}
            try:
                parsed = jsonlib.loads(m.group(0))
            except Exception:
                return {"tool": "none", "response": raw}
        return parsed

    def _finalize_with_llm(self, user_message: str, tool_name: str, tool_result: Any, sql_used: Optional[str]) -> str:
        """
        Pide al LLM que redacte la respuesta final (incluye código de gráficos solo si el usuario lo pidió explícitamente).
        """
        wants_chart = self._wants_chart(user_message)
        allow_chart = wants_chart if self.valves.REQUIRE_EXPLICIT_CHART_REQUEST else True

        system = (
            "Redacta una respuesta clara y accionable. "
            + ("Incluye un bloque de código Python (matplotlib o seaborn) porque el usuario lo pidió. " if allow_chart else "No incluyas gráficos ni código de gráficos a menos que el usuario lo pida explícitamente. ")
            + "No inventes columnas ni datos. Cuando resumas resultados tabulares, evita verbosidad innecesaria."
        )
        context = {
            "tool": tool_name,
            "sql": sql_used,
            "result_preview": tool_result if isinstance(tool_result, (dict, list)) else str(tool_result),
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
            {"role": "system", "content": f"Contexto de herramienta: {jsonlib.dumps(context)[: self.valves.SCHEMA_MAX_CHARS]}"},
        ]
        data = self._ollama_chat(messages, temperature=0.3)
        return data["choices"][0]["message"]["content"].strip()

    def _wants_chart(self, text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "gráfico", "grafico", "gráfica", "grafica", "plot", "chart", "visualiza", "visualización", "visualizacion",
            "matplotlib", "seaborn", "plotly"
        ]
        return any(k in t for k in keywords)

    # =====================
    # Ejecución de consulta
    # =====================
    def _execute_sql(self, sql: str) -> Dict[str, Any]:
        url = self._compose_url(self.valves.SQL_ENDPOINT_PATH)
        payload = {"sql": sql}
        r = requests.post(url, headers=self._build_headers(), json=payload, timeout=120)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}

    # =====================
    # Formateo de resultados
    # =====================
    def _results_to_markdown(self, result: Dict[str, Any], max_rows: int = 50) -> str:
        # Intentar manejar dos formatos: {columns: [...], rows: [[...], ...]} o List[Dict]
        headers: List[str] = []
        rows: List[List[Any]] = []

        if isinstance(result, dict) and "columns" in result and "rows" in result:
            headers = [str(c) for c in result.get("columns", [])]
            for row in result.get("rows", [])[:max_rows]:
                rows.append([row[i] if i < len(row) else None for i in range(len(headers))])
        elif isinstance(result, list):
            # lista de objetos
            all_keys = set()
            for item in result:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            headers = list(all_keys)
            for item in result[:max_rows]:
                if isinstance(item, dict):
                    rows.append([item.get(h) for h in headers])
        elif isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
            return self._results_to_markdown(result["data"], max_rows=max_rows)
        else:
            return f"```\n{jsonlib.dumps(result, indent=2, ensure_ascii=False)}\n```"

        if not headers:
            return "(Sin resultados)"

        # Construir tabla Markdown simple
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            md += "| " + " | ".join([self._to_str(cell) for cell in row]) + " |\n"
        return md

    def _to_str(self, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            try:
                dp = max(0, int(self.valves.DECIMAL_PLACES))
                fmt = "{:0." + str(dp) + "f}"
                return fmt.format(v)
            except Exception:
                return str(v)
        s = str(v)
        return s.replace("\n", " ")[:500]

    # =====================
    # Detección de intención
    # =====================
    def _is_sales_intent(self, user_message: str) -> bool:
        text = user_message.lower()
        return any(kw in text for kw in self.valves.SALES_KEYWORDS) and (
            "usa" in text or "estados unidos" in text or "us " in text or text.endswith(" us")
        )

    # =====================
    # Entrada principal
    # =====================
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Flujo:
        1) Detectar intención de ventas QuickBooks USA
        2) Obtener esquema (con cache)
        3) (Opcional) Refrescar datos
        4) Generar SQL con Ollama
        5) Ejecutar SQL y devolver resultados
        """

        is_stream = bool(body.get("stream", True))

        # Conversación general con tool-calling: si el LLM decide usar SQL, se ejecuta; si no, responde directo

        if is_stream:
            def stream() -> Iterator[str]:
                try:
                    schema = None
                    try:
                        schema = self._get_schema(force=False)
                    except Exception:
                        pass

                    if self.valves.TOOLS_ENABLED:
                        yield "🧭 Decidiendo si usar herramientas..."
                        decision = self._decide_tool(user_message, schema)
                    else:
                        decision = {"tool": "none"}

                    tool = decision.get("tool", "none")

                    if tool == "refresh_data":
                        yield "🔄 Refrescando datos desde QuickBooks Online..."
                        try:
                            _ = self._refresh_data()
                            yield "✅ Refresh solicitado"
                            final = self._finalize_with_llm(user_message, tool, {"status": "ok"}, None)
                            yield final
                        except Exception as e:
                            yield f"❌ Error en refresh: {e}"
                        return

                    if tool == "get_schema":
                        if schema is None:
                            try:
                                schema = self._get_schema(force=True)
                            except Exception as e:
                                yield f"❌ Error obteniendo esquema: {e}"
                                return
                        preview = jsonlib.dumps(schema)[:1000]
                        final = self._finalize_with_llm(user_message, tool, {"schema_preview": preview}, None)
                        yield final
                        return

                    if tool == "execute_sql":
                        sql = decision.get("sql", "").strip()
                        if not sql:
                            yield "❌ No se proporcionó SQL en la decisión de herramienta"
                            return
                        try:
                            self._validate_sql(sql)
                            if self.valves.ENFORCE_COUNTRY_FILTER:
                                sql = self._ensure_country_filter(sql)
                            sql = self._ensure_limit(sql)
                        except Exception as e:
                            yield f"❌ SQL inválido: {e}"
                            return

                        yield f"▶️ Ejecutando consulta...\n```sql\n{sql}\n```"
                        try:
                            result = self._execute_sql(sql)
                        except Exception as e:
                            yield f"❌ Error ejecutando SQL: {e}"
                            return

                        md = self._results_to_markdown(result, max_rows=50)
                        yield "✅ Resultados (previa):\n" + md
                        final = self._finalize_with_llm(user_message, tool, result, sql)
                        yield final
                        return

                    # tool == none -> respuesta directa del LLM
                    if self.valves.ALLOW_GENERAL_CHAT:
                        yield "💬 Respondiendo directamente sin herramientas..."
                        direct = self._finalize_with_llm(user_message, "none", {}, None)
                        yield direct
                    else:
                        yield "ℹ️ Ninguna herramienta aplicada y chat general deshabilitado."

                except Exception as e:
                    yield f"❌ Error en el pipeline: {e}"

            return stream()
        else:
            try:
                schema = None
                try:
                    schema = self._get_schema(force=False)
                except Exception:
                    pass

                decision = self._decide_tool(user_message, schema) if self.valves.TOOLS_ENABLED else {"tool": "none"}
                tool = decision.get("tool", "none")

                if tool == "refresh_data":
                    try:
                        _ = self._refresh_data()
                        final = self._finalize_with_llm(user_message, tool, {"status": "ok"}, None)
                        return {"choices": [{"message": {"content": final}}]}
                    except Exception as e:
                        return {"choices": [{"message": {"content": f"Error en refresh: {e}"}}]}

                if tool == "get_schema":
                    if schema is None:
                        try:
                            schema = self._get_schema(force=True)
                        except Exception as e:
                            return {"choices": [{"message": {"content": f"Error obteniendo esquema: {e}"}}]}
                    preview = jsonlib.dumps(schema)[:1000]
                    final = self._finalize_with_llm(user_message, tool, {"schema_preview": preview}, None)
                    return {"choices": [{"message": {"content": final}}]}

                if tool == "execute_sql":
                    sql = decision.get("sql", "").strip()
                    if not sql:
                        return {"choices": [{"message": {"content": "Error: No se proporcionó SQL"}}]}
                    try:
                        self._validate_sql(sql)
                        if self.valves.ENFORCE_COUNTRY_FILTER:
                            sql = self._ensure_country_filter(sql)
                        sql = self._ensure_limit(sql)
                        result = self._execute_sql(sql)
                        md = self._results_to_markdown(result, max_rows=50)
                        summary = f"SQL ejecutado:\n```sql\n{sql}\n```\n\n{md}"
                        final = self._finalize_with_llm(user_message, tool, result, sql)
                        return {"choices": [{"message": {"content": summary + "\n\n" + final}}]}
                    except Exception as e:
                        return {"choices": [{"message": {"content": f"Error ejecutando SQL: {e}"}}]}

                # tool == none -> respuesta directa del LLM
                if self.valves.ALLOW_GENERAL_CHAT:
                    direct = self._finalize_with_llm(user_message, "none", {}, None)
                    return {"choices": [{"message": {"content": direct}}]}
                else:
                    return {"choices": [{"message": {"content": "Ninguna herramienta aplicada y chat general deshabilitado."}}]}
            except Exception as e:
                return {"choices": [{"message": {"content": f"Error: {e}"}}]}


