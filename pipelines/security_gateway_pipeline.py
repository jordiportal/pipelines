"""
title: Security Gateway Pipeline
author: security-team
date: 2024-01-30
version: 1.0
license: MIT
description: Intelligent security gateway that routes queries to local or external LLM based on confidentiality analysis
requirements: requests, pydantic
"""

import os
import json
import requests
import time
import uuid
from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Import ComfyUI Pipeline
try:
    import sys
    sys.path.append('.')
    from pipelines.comfyui_image_generator_pipeline import Pipeline as ComfyUIPipeline
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("⚠️ ComfyUI Pipeline no disponible - detección de imágenes deshabilitada")


class Pipeline:
    class Valves(BaseModel):
        # === Ollama Configuration (Local LLM for Security Analysis) ===
        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434", 
            description="URL del servidor Ollama local"
        )
        OLLAMA_SECURITY_MODEL: str = Field(
            default="llama3.1:8b", 
            description="Modelo Ollama para análisis de seguridad (ej: llama3.1:8b, mistral:7b)"
        )
        OLLAMA_RESPONSE_MODEL: str = Field(
            default="llama3.1:70b", 
            description="Modelo Ollama para respuestas con datos confidenciales"
        )
        
        # === OpenAI Configuration (External LLM for Non-Confidential) ===
        OPENAI_API_KEY: str = Field(
            default="", 
            description="Clave API de OpenAI para datos no confidenciales"
        )
        OPENAI_MODEL: str = Field(
            default="gpt-4", 
            description="Modelo OpenAI para datos no confidenciales (ej: gpt-4, gpt-3.5-turbo)"
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1", 
            description="Base URL de OpenAI API"
        )
        
        # === Security Configuration ===
        SECURITY_PROMPT: str = Field(
            default="""Analiza si esta consulta contiene información confidencial empresarial que no debería procesarse con servicios externos.

CRITERIOS PARA CONSIDERAR CONFIDENCIAL:
- Datos financieros específicos (ventas, ingresos, costos, presupuestos)
- Información personal de clientes o empleados
- Estrategias empresariales internas o secretos comerciales
- Datos operativos sensibles (procesos internos, métricas de negocio)
- Información contractual, legal o regulatoria
- Cualquier dato que podría comprometer la ventaja competitiva

CRITERIOS PARA CONSIDERAR NO CONFIDENCIAL:
- Preguntas generales sobre metodologías o mejores prácticas
- Consultas educativas o de conocimiento general
- Solicitudes de ayuda técnica sin datos específicos de la empresa
- Preguntas sobre herramientas o tecnologías públicas""", 
            description="Prompt libre para que el LLM determine qué es confidencial"
        )
        SECURITY_THRESHOLD: float = Field(
            default=0.6, 
            description="Umbral de confianza para clasificar como confidencial (0.0-1.0)"
        )
        LOG_SECURITY_DECISIONS: bool = Field(
            default=True, 
            description="Registrar decisiones de seguridad en logs"
        )
        
        # === ComfyUI Integration ===
        COMFYUI_PIPELINE_ENABLED: bool = Field(
            default=True,
            description="Habilitar integración con pipeline ComfyUI para generación de imágenes"
        )
        COMFYUI_BASE_URL: str = Field(
            default="http://192.168.7.101:8188",
            description="URL base del servidor ComfyUI"
        )
        
        # === Advanced Settings ===
        ENABLE_STREAMING: bool = Field(
            default=True, 
            description="Habilitar streaming de respuestas"
        )
        OLLAMA_TIMEOUT: int = Field(
            default=60, 
            description="Timeout en segundos para conexiones Ollama"
        )
        OPENAI_TIMEOUT: int = Field(
            default=60, 
            description="Timeout en segundos para conexiones OpenAI"
        )

    def __init__(self):
        self.name = "Security Gateway"
        self.valves = self.Valves()
        self._security_log = []
        
        # Initialize ComfyUI pipeline if available
        self.comfyui_pipeline = None
        if COMFYUI_AVAILABLE and self.valves.COMFYUI_PIPELINE_ENABLED:
            try:
                self.comfyui_pipeline = ComfyUIPipeline()
                print("✅ ComfyUI Pipeline cargado correctamente")
            except Exception as e:
                print(f"❌ Error cargando ComfyUI Pipeline: {e}")
                self.comfyui_pipeline = None
        
    async def on_startup(self):
        """Initialize and test connections"""
        print(f"🛡️ Iniciando Security Gateway...")
        
        # Test Ollama connection
        await self._test_ollama_connection()
        
        # Test OpenAI connection (if API key provided)
        if self.valves.OPENAI_API_KEY:
            await self._test_openai_connection()
        else:
            print("⚠️ OpenAI API key no configurada - solo se usará Ollama local")
            
        print(f"✅ Security Gateway iniciado correctamente")
        
    async def on_shutdown(self):
        print(f"🛡️ Cerrando Security Gateway...")
        
        # Save security log if enabled
        if self.valves.LOG_SECURITY_DECISIONS and self._security_log:
            self._save_security_log()

    async def _test_ollama_connection(self):
        """Test Ollama server connectivity"""
        try:
            response = requests.get(
                f"{self.valves.OLLAMA_BASE_URL}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                print(f"✅ Ollama conectado - Modelos disponibles: {model_names}")
                
                # Check if required models are available
                if self.valves.OLLAMA_SECURITY_MODEL not in model_names:
                    print(f"⚠️ Modelo de seguridad '{self.valves.OLLAMA_SECURITY_MODEL}' no encontrado")
                if self.valves.OLLAMA_RESPONSE_MODEL not in model_names:
                    print(f"⚠️ Modelo de respuesta '{self.valves.OLLAMA_RESPONSE_MODEL}' no encontrado")
            else:
                print(f"❌ Error conectando a Ollama: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Error conectando a Ollama: {e}")

    async def _test_openai_connection(self):
        """Test OpenAI API connectivity"""
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                f"{self.valves.OPENAI_BASE_URL}/models",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                print("✅ OpenAI API conectada correctamente")
            else:
                print(f"❌ Error conectando a OpenAI: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Error conectando a OpenAI: {e}")

    def _log_security_decision(self, user_message: str, is_confidential: bool, confidence: float, reasoning: str):
        """Log security classification decisions"""
        if self.valves.LOG_SECURITY_DECISIONS:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "message_hash": hash(user_message) % 10000,  # Hash for privacy
                "is_confidential": is_confidential,
                "confidence": confidence,
                "reasoning": reasoning,
                "llm_used": "ollama_local" if is_confidential else "openai_external"
            }
            self._security_log.append(log_entry)
            
            # Print to console for immediate visibility
            status = "🔒 CONFIDENCIAL" if is_confidential else "🌐 NO CONFIDENCIAL"
            print(f"🛡️ {status} (confianza: {confidence:.2f}) -> {'Local' if is_confidential else 'Externo'}")

    def _log_image_generation_decision(self, user_message: str):
        """Log image generation routing decisions"""
        if self.valves.LOG_SECURITY_DECISIONS:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "message_hash": hash(user_message) % 10000,  # Hash for privacy
                "request_type": "image_generation",
                "pipeline_used": "comfyui",
                "reasoning": "Solicitud de generación de imagen detectada automáticamente"
            }
            self._security_log.append(log_entry)
            
            # Print to console for immediate visibility
            print(f"🎨 IMAGEN DETECTADA -> ComfyUI")

    def _save_security_log(self):
        """Save security log to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_decisions_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self._security_log, f, indent=2, ensure_ascii=False)
            print(f"📝 Log de seguridad guardado: {filename}")
        except Exception as e:
            print(f"❌ Error guardando log de seguridad: {e}")

    def _analyze_confidentiality(self, user_message: str) -> Dict[str, Any]:
        """Analyze if the message contains confidential information using local Ollama"""
        
        # Create analysis prompt using the configurable security prompt
        analysis_prompt = f"""{self.valves.SECURITY_PROMPT}

CONSULTA A ANALIZAR: "{user_message}"

Responde ÚNICAMENTE en formato JSON:
{{
    "es_confidencial": true/false,
    "confianza": 0.0-1.0,
    "razonamiento": "explicación breve de por qué es o no confidencial según los criterios",
    "categoria": "tipo de datos si es confidencial (ej: financiero, personal, estratégico)"
}}"""

        try:
            # Call Ollama for security analysis
            response = requests.post(
                f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": self.valves.OLLAMA_SECURITY_MODEL,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "Eres un analista de seguridad experto. Analiza consultas para determinar si contienen información confidencial empresarial. Responde solo en JSON válido."
                        },
                        {"role": "user", "content": analysis_prompt}
                    ],
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=self.valves.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON response
                try:
                    analysis = json.loads(content)
                    return {
                        "is_confidential": analysis.get("es_confidencial", False),
                        "confidence": analysis.get("confianza", 0.5),
                        "reasoning": analysis.get("razonamiento", "Análisis completado"),
                        "category": analysis.get("categoria", "general")
                    }
                except json.JSONDecodeError:
                    # Fallback to conservative analysis
                    print(f"⚠️ Error parsing Ollama JSON response: {content}")
                    return self._fallback_conservative_analysis(user_message)
            else:
                print(f"❌ Error en análisis Ollama: HTTP {response.status_code}")
                return self._fallback_conservative_analysis(user_message)
                
        except Exception as e:
            print(f"❌ Error conectando a Ollama para análisis: {e}")
            return self._fallback_conservative_analysis(user_message)

    def _fallback_conservative_analysis(self, user_message: str) -> Dict[str, Any]:
        """Conservative fallback analysis when Ollama fails - assumes confidential for safety"""
        # When in doubt, err on the side of caution and treat as confidential
        return {
            "is_confidential": True,
            "confidence": 0.7,  # High confidence in safety decision
            "reasoning": "Análisis de seguridad falló - aplicando principio de precaución (tratando como confidencial)",
            "category": "fallback-safe"
        }

    def _is_image_request(self, user_message: str) -> bool:
        """Detecta si el mensaje es una solicitud de generación de imagen"""
        image_keywords = [
            "genera", "crea", "dibuja", "imagen", "picture", "image", 
            "draw", "create", "generate", "pintura", "dibujo", "foto",
            "ilustra", "diseña", "render", "visualiza"
        ]
        
        # Buscar palabras clave de imagen
        message_lower = user_message.lower()
        has_image_keyword = any(keyword in message_lower for keyword in image_keywords)
        
        # Patrones adicionales
        image_patterns = [
            "una imagen de",
            "una foto de", 
            "un dibujo de",
            "una ilustración de",
            "que se vea como",
            "mostrar como imagen"
        ]
        
        has_image_pattern = any(pattern in message_lower for pattern in image_patterns)
        
        return has_image_keyword or has_image_pattern

    def _call_comfyui_pipeline(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Llama al pipeline ComfyUI para generar imágenes"""
        if not self.comfyui_pipeline:
            return "❌ Pipeline ComfyUI no disponible. Verifique la configuración."
        
        try:
            print("🎨 Enrutando a ComfyUI para generación de imagen...")
            
            # Agregar mensaje informativo
            info_msg = "🎨 **Solicitud de imagen detectada**\n→ Enrutando a ComfyUI para generación\n\n"
            
            if body.get("stream", False):
                def image_stream():
                    # Primero enviar mensaje informativo
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': info_msg}}]})}\n\n"
                    
                    # Llamar al pipeline ComfyUI
                    try:
                        result = self.comfyui_pipeline.pipe(user_message, model_id, messages, body)
                        
                        if hasattr(result, '__iter__') and not isinstance(result, str):
                            # Es un generador/iterador
                            for chunk in result:
                                if isinstance(chunk, str):
                                    # Convertir string a formato de streaming
                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                                else:
                                    yield chunk
                        else:
                            # Es una respuesta simple
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': str(result)}}]})}\n\n"
                    
                    except Exception as e:
                        error_msg = f"❌ Error en generación de imagen: {str(e)}"
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': error_msg}}]})}\n\n"
                
                return image_stream()
            else:
                # Non-streaming
                try:
                    result = self.comfyui_pipeline.pipe(user_message, model_id, messages, body)
                    
                    if isinstance(result, str):
                        return info_msg + result
                    elif hasattr(result, '__iter__'):
                        # Convertir generador a string
                        content_parts = [info_msg]
                        for chunk in result:
                            if isinstance(chunk, str):
                                content_parts.append(chunk)
                        return "".join(content_parts)
                    else:
                        return info_msg + str(result)
                        
                except Exception as e:
                    return info_msg + f"❌ Error en generación de imagen: {str(e)}"
                    
        except Exception as e:
            return f"❌ Error llamando pipeline ComfyUI: {str(e)}"

    def _call_ollama(self, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Call local Ollama for confidential data"""
        try:
            # Prepare payload for Ollama
            ollama_payload = {
                **body,
                "model": self.valves.OLLAMA_RESPONSE_MODEL
            }
            
            # Clean payload
            if "user" in ollama_payload:
                del ollama_payload["user"]
            if "chat_id" in ollama_payload:
                del ollama_payload["chat_id"]
            if "title" in ollama_payload:
                del ollama_payload["title"]
            
            response = requests.post(
                f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json=ollama_payload,
                stream=body.get("stream", False),
                timeout=self.valves.OLLAMA_TIMEOUT
            )
            
            response.raise_for_status()
            
            if body.get("stream", False):
                return response.iter_lines()
            else:
                return response.json()
                
        except Exception as e:
            return f"❌ Error en Ollama local: {e}"

    def _call_openai(self, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Call external OpenAI for non-confidential data"""
        if not self.valves.OPENAI_API_KEY:
            return "❌ OpenAI API key no configurada. Configure la clave para datos no confidenciales."
        
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload for OpenAI
            openai_payload = {
                **body,
                "model": self.valves.OPENAI_MODEL
            }
            
            # Clean payload
            if "user" in openai_payload:
                del openai_payload["user"]
            if "chat_id" in openai_payload:
                del openai_payload["chat_id"]
            if "title" in openai_payload:
                del openai_payload["title"]
            
            response = requests.post(
                f"{self.valves.OPENAI_BASE_URL}/chat/completions",
                json=openai_payload,
                headers=headers,
                stream=body.get("stream", False),
                timeout=self.valves.OPENAI_TIMEOUT
            )
            
            response.raise_for_status()
            
            if body.get("stream", False):
                return response.iter_lines()
            else:
                return response.json()
                
        except Exception as e:
            return f"❌ Error en OpenAI externo: {e}"

    def _create_security_info_message(self, is_confidential: bool, confidence: float, reasoning: str) -> str:
        """Create informational message about security routing"""
        if is_confidential:
            return f"🔒 **Datos confidenciales detectados** (confianza: {confidence:.0%})\n" \
                   f"→ Procesando con LLM local por seguridad\n" \
                   f"→ Razón: {reasoning}\n\n"
        else:
            return f"🌐 **Datos no confidenciales** (confianza: {confidence:.0%})\n" \
                   f"→ Procesando con LLM externo optimizado\n" \
                   f"→ Razón: {reasoning}\n\n"

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline with security-based routing and image detection"""
        
        # Step 0: Check for image generation requests
        if self.valves.COMFYUI_PIPELINE_ENABLED and self._is_image_request(user_message):
            print(f"🎨 Solicitud de imagen detectada - enrutando a ComfyUI...")
            
            # Log image generation decision
            self._log_image_generation_decision(user_message)
            
            return self._call_comfyui_pipeline(user_message, model_id, messages, body)
        
        # Step 1: Analyze confidentiality
        print(f"🛡️ Analizando confidencialidad de la consulta...")
        
        analysis = self._analyze_confidentiality(user_message)
        is_confidential = analysis["is_confidential"]
        confidence = analysis["confidence"]
        reasoning = analysis["reasoning"]
        
        # Step 2: Make routing decision
        route_to_local = is_confidential or confidence >= self.valves.SECURITY_THRESHOLD
        
        # Step 3: Log the decision
        self._log_security_decision(user_message, route_to_local, confidence, reasoning)
        
        # Step 4: Route to appropriate LLM
        if route_to_local:
            print(f"🔒 Enrutando a Ollama local para datos confidenciales...")
            
            if body.get("stream", False):
                def local_stream():
                    # Stream security info first
                    security_msg = self._create_security_info_message(True, confidence, reasoning)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': security_msg}}]})}\n\n"
                    
                    # Then stream the actual response
                    response_stream = self._call_ollama(messages, body)
                    if isinstance(response_stream, str):
                        # Error case
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': response_stream}}]})}\n\n"
                    else:
                        # Stream response
                        for line in response_stream:
                            if line:
                                yield line
                
                return local_stream()
            else:
                # Non-streaming response
                security_msg = self._create_security_info_message(True, confidence, reasoning)
                llm_response = self._call_ollama(messages, body)
                
                if isinstance(llm_response, str):
                    return security_msg + llm_response
                elif isinstance(llm_response, dict):
                    # Modify the response content
                    if "choices" in llm_response and len(llm_response["choices"]) > 0:
                        original_content = llm_response["choices"][0]["message"]["content"]
                        llm_response["choices"][0]["message"]["content"] = security_msg + original_content
                    return llm_response
                else:
                    return security_msg + str(llm_response)
        
        else:
            print(f"🌐 Enrutando a OpenAI externo para datos no confidenciales...")
            
            if body.get("stream", False):
                def external_stream():
                    # Stream security info first
                    security_msg = self._create_security_info_message(False, confidence, reasoning)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': security_msg}}]})}\n\n"
                    
                    # Then stream the actual response
                    response_stream = self._call_openai(messages, body)
                    if isinstance(response_stream, str):
                        # Error case
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': response_stream}}]})}\n\n"
                    else:
                        # Stream response
                        for line in response_stream:
                            if line:
                                yield line
                
                return external_stream()
            else:
                # Non-streaming response
                security_msg = self._create_security_info_message(False, confidence, reasoning)
                llm_response = self._call_openai(messages, body)
                
                if isinstance(llm_response, str):
                    return security_msg + llm_response
                elif isinstance(llm_response, dict):
                    # Modify the response content
                    if "choices" in llm_response and len(llm_response["choices"]) > 0:
                        original_content = llm_response["choices"][0]["message"]["content"]
                        llm_response["choices"][0]["message"]["content"] = security_msg + original_content
                    return llm_response
                else:
                    return security_msg + str(llm_response)
