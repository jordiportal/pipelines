"""
title: ComfyUI Image Generator Pipeline
author: assistant
date: 2024-12-19
version: 1.0
license: MIT
description: Pipeline que se conecta con ComfyUI para generar imágenes usando OmniGen2, incluye mejora de prompts y actualizaciones de estado en tiempo real.
requirements: requests, pillow
"""

import json
import time
import uuid
import random
import base64
import io
import os
from pathlib import Path
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import requests

from schemas import OpenAIChatMessage


class Pipeline:
    class Valves(BaseModel):
        # === Configuración del servidor ComfyUI ===
        COMFYUI_BASE_URL: str = Field(
            default="http://192.168.7.101:8188", 
            description="URL completa del servidor ComfyUI (ej: http://192.168.7.101:8188)"
        )
        
        # === Configuración de generación de imágenes ===
        WIDTH: int = Field(
            default=1024, 
            description="Ancho de la imagen generada en píxeles"
        )
        HEIGHT: int = Field(
            default=1024, 
            description="Alto de la imagen generada en píxeles"
        )
        STEPS: int = Field(
            default=20, 
            description="Número de pasos de difusión (más pasos = mejor calidad, más tiempo)"
        )
        CFG_SCALE: float = Field(
            default=5.0, 
            description="Escala CFG para seguir el prompt (1.0-20.0, recomendado: 5.0-7.0)"
        )
        
        # === Configuración de prompts ===
        NEGATIVE_PROMPT: str = Field(
            default="blurry, low quality, distorted, ugly, bad anatomy, deformed, poorly drawn, text, watermark, signature",
            description="Prompt negativo para evitar elementos no deseados"
        )
        ENHANCE_PROMPTS: bool = Field(
            default=True, 
            description="Activar mejora automática de prompts para mejores resultados"
        )
        PROMPT_ENHANCEMENT_STYLE: str = Field(
            default="detailed", 
            description="Estilo de mejora: detailed, artistic, photorealistic, fantasy"
        )
        
        # === Modelos ComfyUI ===
        UNET_MODEL: str = Field(
            default="omnigen2_fp16.safetensors",
            description="Modelo UNet para generación (debe existir en ComfyUI/models/unet/)"
        )
        VAE_MODEL: str = Field(
            default="ae.safetensors",
            description="Modelo VAE para decodificación (debe existir en ComfyUI/models/vae/)"
        )
        CLIP_MODEL: str = Field(
            default="qwen_2.5_vl_fp16.safetensors",
            description="Modelo CLIP para codificación de texto (debe existir en ComfyUI/models/clip/)"
        )
        




    def __init__(self):
        # Identificador único del pipeline
        self.id = "comfyui_image_generator"
        self.name = "ComfyUI Image Generator"
        self.valves = self.Valves()
        self.client_id = str(uuid.uuid4())
        
        # Flag para evitar múltiples generaciones
        self._generating = False
        
        # Cargar template de workflow
        self.workflow_template = self._load_workflow_template()

    def _load_workflow_template(self):
        """Carga el template de workflow de ComfyUI desde el archivo JSON"""
        try:
            with open("templates/comfyui/image_omnigen2_t2i.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # Template por defecto si no existe el archivo
            return self._get_default_workflow()

    def _get_default_workflow(self):
        """Template de workflow por defecto para OmniGen2"""
        return {
            "6": {
                "inputs": {
                    "text": "",
                    "clip": ["10", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Codificar Texto CLIP (Prompt)"}
            },
            "7": {
                "inputs": {
                    "text": "",
                    "clip": ["10", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Codificar Texto CLIP (Prompt Negativo)"}
            },
            "8": {
                "inputs": {
                    "samples": ["28", 0],
                    "vae": ["13", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "Decodificación VAE"}
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Guardar Imagen"}
            },
            "10": {
                "inputs": {
                    "clip_name": self.valves.CLIP_MODEL,
                    "type": "omnigen2",
                    "device": "default"
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Cargar CLIP"}
            },
            "11": {
                "inputs": {
                    "width": self.valves.WIDTH,
                    "height": self.valves.HEIGHT,
                    "batch_size": 1
                },
                "class_type": "EmptySD3LatentImage",
                "_meta": {"title": "EmptySD3LatentImage"}
            },
            "12": {
                "inputs": {
                    "unet_name": self.valves.UNET_MODEL,
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Cargar Modelo de Difusión"}
            },
            "13": {
                "inputs": {
                    "vae_name": self.valves.VAE_MODEL
                },
                "class_type": "VAELoader",
                "_meta": {"title": "Cargar VAE"}
            },
            "20": {
                "inputs": {
                    "sampler_name": "euler"
                },
                "class_type": "KSamplerSelect",
                "_meta": {"title": "KSamplerSelect"}
            },
            "21": {
                "inputs": {
                    "noise_seed": random.randint(1, 2**32)
                },
                "class_type": "RandomNoise",
                "_meta": {"title": "Ruido aleatorio"}
            },
            "23": {
                "inputs": {
                    "scheduler": "simple",
                    "steps": self.valves.STEPS,
                    "denoise": 1,
                    "model": ["12", 0]
                },
                "class_type": "BasicScheduler",
                "_meta": {"title": "ProgramadorBásico"}
            },
            "27": {
                "inputs": {
                    "cfg_conds": self.valves.CFG_SCALE,
                    "cfg_cond2_negative": 2,
                    "style": "regular",
                    "model": ["12", 0],
                    "cond1": ["6", 0],
                    "cond2": ["7", 0],
                    "negative": ["7", 0]
                },
                "class_type": "DualCFGGuider",
                "_meta": {"title": "Guía Dual CFG"}
            },
            "28": {
                "inputs": {
                    "noise": ["21", 0],
                    "guider": ["27", 0],
                    "sampler": ["20", 0],
                    "sigmas": ["23", 0],
                    "latent_image": ["11", 0]
                },
                "class_type": "SamplerCustomAdvanced",
                "_meta": {"title": "SamplerCustomAdvanced"}
            }
        }

    def _enhance_prompt(self, prompt: str) -> str:
        """Mejora el prompt para obtener mejores resultados"""
        if not self.valves.ENHANCE_PROMPTS:
            return prompt
            
        enhancement_styles = {
            "detailed": "highly detailed, professional quality, sharp focus, vibrant colors, excellent composition",
            "artistic": "artistic masterpiece, creative composition, beautiful lighting, expressive style",
            "photorealistic": "photorealistic, ultra-realistic, high resolution, professional photography, perfect lighting",
            "fantasy": "fantasy art style, magical atmosphere, enchanting details, mystical lighting, epic composition"
        }
        
        enhancement = enhancement_styles.get(self.valves.PROMPT_ENHANCEMENT_STYLE, enhancement_styles["detailed"])
        
        # Agregar mejoras al prompt
        enhanced_prompt = f"{prompt}, {enhancement}"
        
        return enhanced_prompt

    def _prepare_workflow(self, prompt: str) -> dict:
        """Prepara el workflow con el prompt y configuración actual"""
        workflow = self.workflow_template.copy()
        
        # Mejorar el prompt
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Actualizar prompt positivo
        workflow["6"]["inputs"]["text"] = enhanced_prompt
        
        # Actualizar prompt negativo
        workflow["7"]["inputs"]["text"] = self.valves.NEGATIVE_PROMPT
        
        # Actualizar configuraciones
        workflow["11"]["inputs"]["width"] = self.valves.WIDTH
        workflow["11"]["inputs"]["height"] = self.valves.HEIGHT
        workflow["23"]["inputs"]["steps"] = self.valves.STEPS
        workflow["27"]["inputs"]["cfg_conds"] = self.valves.CFG_SCALE
        
        # Actualizar modelos
        workflow["10"]["inputs"]["clip_name"] = self.valves.CLIP_MODEL
        workflow["12"]["inputs"]["unet_name"] = self.valves.UNET_MODEL
        workflow["13"]["inputs"]["vae_name"] = self.valves.VAE_MODEL
        
        # Generar nueva semilla aleatoria
        workflow["21"]["inputs"]["noise_seed"] = random.randint(1, 2**32)
        
        return workflow

    def _queue_prompt(self, workflow: dict) -> str:
        """Envía el workflow a ComfyUI y retorna el prompt_id"""
        prompt_data = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        url = f"{self.valves.COMFYUI_BASE_URL}/prompt"
        
        try:
            response = requests.post(url, json=prompt_data, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result["prompt_id"]
        except Exception as e:
            raise Exception(f"Error al enviar prompt a ComfyUI: {str(e)}")



    def _get_image_data_url(self, filename: str, subfolder: str = "", folder_type: str = "output") -> str:
        """Descarga la imagen de ComfyUI y la convierte a data URL (sin guardar archivo)"""
        url = f"{self.valves.COMFYUI_BASE_URL}/view"
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Verificar contenido
            image_data = response.content
            if not image_data:
                raise Exception("Imagen vacía recibida de ComfyUI")
            
            # Comprimir imagen si es muy grande (>800KB)
            if len(image_data) > 800 * 1024:  # > 800KB
                try:
                    compressed_data = self._compress_image_bytes(image_data)
                    image_data = compressed_data
                except Exception:
                    # Si falla la compresión, usar imagen original
                    pass
            
            # Crear data URL
            content_type = response.headers.get('content-type', 'image/png')
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            data_url = f"data:{content_type};base64,{image_base64}"
            
            return data_url
            
        except Exception as e:
            raise Exception(f"Error descargando imagen de ComfyUI: {str(e)}")

    def _compress_image_bytes(self, image_data: bytes) -> bytes:
        """Comprime datos de imagen para reducir tamaño"""
        try:
            from PIL import Image
            import io
            
            # Abrir imagen
            img = Image.open(io.BytesIO(image_data))
            
            # Redimensionar si es muy grande
            if img.width > 768 or img.height > 768:
                img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            
            # Convertir a RGB si tiene transparencia
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Comprimir como JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=75, optimize=True)
            return output.getvalue()
            
        except ImportError:
            # Sin PIL, devolver imagen original
            return image_data
        except Exception:
            # Si hay error en la compresión, usar imagen original
            return image_data

    def _monitor_progress(self, prompt_id: str) -> Generator[str, None, tuple]:
        """Monitorea el progreso de generación usando polling"""
        yield "🔄 Monitoreando progreso con polling..."
        
        # Usar polling directamente (más simple y confiable)
        return self._poll_for_completion(prompt_id)

    def _poll_for_completion(self, prompt_id: str) -> tuple:
        """Polling para verificar completitud con actualizaciones de progreso"""
        url = f"{self.valves.COMFYUI_BASE_URL}/history/{prompt_id}"
        
        timeout = 300  # 5 minutos
        start_time = time.time()
        last_update = 0
        
        while (time.time() - start_time) < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id]["outputs"]
                        if "9" in outputs and "images" in outputs["9"]:
                            image_info = outputs["9"]["images"][0]
                            return image_info["filename"], image_info.get("subfolder", "")
                
                # Actualizar progreso cada 10 segundos
                elapsed = time.time() - start_time
                if elapsed - last_update >= 10:
                    estimated_progress = min(int((elapsed / 60) * 100), 95)  # Estimar basado en tiempo
                    print(f"⏳ Procesando... {int(elapsed)}s transcurridos (~{estimated_progress}%)")
                    last_update = elapsed
                
                time.sleep(2)
                
            except Exception:
                time.sleep(2)
                continue
        
        raise Exception("Timeout esperando completitud de imagen")

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        print(f"ComfyUI Server: {self.valves.COMFYUI_BASE_URL}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        """Se ejecuta cuando las valves son actualizadas"""
        pass

    def _extract_user_message(self, message: str) -> str:
        """Extrae el mensaje original del usuario del historial de chat si es necesario"""
        # Si el mensaje contiene chat_history, extraer el último mensaje del usuario
        if "chat_history" in message.lower() and "<chat_history>" in message:
            try:
                import re
                user_messages = re.findall(r'USER:\s*(.+?)(?=\nASSISTANT:|$)', message, re.IGNORECASE | re.DOTALL)
                if user_messages:
                    return user_messages[-1].strip()
            except Exception:
                # Si hay error, usar mensaje original
                pass
        
        return message

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Pipeline principal para generar imágenes con ComfyUI"""
        
        # Extraer el mensaje original del usuario del historial si es necesario
        original_user_message = self._extract_user_message(user_message)
        
        # Verificar si es una solicitud de generación de imagen
        image_keywords = ["genera", "crea", "dibuja", "imagen", "picture", "image", "draw", "create", "generate"]
        is_image_request = any(keyword in original_user_message.lower() for keyword in image_keywords)
        
        if not is_image_request:
            return "Por favor, solicita la generación de una imagen usando palabras como 'genera', 'crea' o 'dibuja una imagen de...'."

        # Evitar múltiples generaciones simultáneas
        if self._generating:
            return "⚠️ Ya hay una generación de imagen en progreso. Espera a que termine."

        def generate_image():
            try:
                # Marcar como generando
                self._generating = True

                # Extraer el prompt de la descripción
                prompt = original_user_message
                for keyword in ["genera una imagen de", "crea una imagen de", "dibuja", "genera", "crea"]:
                    if keyword in prompt.lower():
                        prompt = prompt.lower().replace(keyword, "").strip()
                        break
                
                if not prompt:
                    yield "❌ No se pudo extraer una descripción válida para la imagen."
                    return
                
                yield f"🚀 Iniciando generación de imagen...\n📝 Prompt: {prompt}"
                
                # Preparar workflow
                yield "⚙️ Preparando workflow de ComfyUI..."
                workflow = self._prepare_workflow(prompt)
                
                # Enviar a cola
                yield "📤 Enviando solicitud a ComfyUI..."
                prompt_id = self._queue_prompt(workflow)
                
                yield f"✅ Solicitud enviada (ID: {prompt_id[:8]}...)"
                
                # Monitorear progreso
                yield "🔄 Monitoreando progreso de generación..."
                
                # Usar polling para monitorear
                filename, subfolder = self._poll_for_completion(prompt_id)
                
                # Descargar imagen
                yield "📥 Descargando imagen de ComfyUI..."
                
                try:
                    # Crear data URL directamente (incluye compresión automática)
                    data_url = self._get_image_data_url(filename, subfolder)
                    
                    # Generar prompt mejorado
                    enhanced_prompt = self._enhance_prompt(prompt)
                    
                    # Emitir mensaje de éxito
                    yield "🎉 ¡Imagen generada exitosamente!"
                    
                    # Mostrar imagen
                    yield f"![Imagen generada]({data_url})"
                    
                    # Emitir detalles
                    details_message = f"""📋 **Detalles de la generación:**
- **Prompt original:** {original_user_message}
- **Prompt mejorado:** {enhanced_prompt}
- **Dimensiones:** {self.valves.WIDTH}x{self.valves.HEIGHT}
- **Pasos:** {self.valves.STEPS}
- **CFG Scale:** {self.valves.CFG_SCALE}
- **Tamaño de imagen:** {len(data_url):,} caracteres"""
                    
                    yield details_message
                    
                    # Finalizar generación exitosa
                    self._generating = False
                    return
                    
                except Exception as img_error:
                    yield f"❌ Error procesando imagen: {str(img_error)}"
                    yield f"📊 Información de debug: filename='{filename}', subfolder='{subfolder}'"
                    # Intentar mostrar al menos los detalles sin imagen
                    enhanced_prompt = self._enhance_prompt(prompt)
                    yield f"""📋 **Detalles de la generación (sin imagen):**
- **Prompt original:** {original_user_message}
- **Prompt mejorado:** {enhanced_prompt}
- **Error:** {str(img_error)}"""
                
            except Exception as e:
                yield f"❌ Error generando imagen: {str(e)}\n\n**Posibles soluciones:**\n- Verificar que ComfyUI esté ejecutándose en {self.valves.COMFYUI_BASE_URL}\n- Comprobar que los modelos estén disponibles\n- Revisar la configuración de valves"
            finally:
                # Siempre limpiar el flag de generación
                self._generating = False

        return generate_image()
