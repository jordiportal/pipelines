# Pipeline de Generación de Imágenes con ComfyUI

Este pipeline permite integrar ComfyUI con Pipelines para generar imágenes usando OmniGen2 directamente desde conversaciones de chat.

## Características

- 🎨 **Generación de imágenes con OmniGen2** usando ComfyUI
- ✨ **Mejora automática de prompts** con diferentes estilos
- 📊 **Actualizaciones de estado en tiempo real** durante la generación
- 🖼️ **Visualización directa** de imágenes en la conversación
- ⚙️ **Configuración flexible** mediante valves
- 🔄 **Monitoreo con WebSocket** y fallback a polling
- 🎯 **Detección inteligente** de solicitudes de imagen

## Requisitos Previos

### 1. ComfyUI Server
Necesitas tener ComfyUI ejecutándose con los siguientes modelos:

- **UNet**: `omnigen2_fp16.safetensors`
- **VAE**: `ae.safetensors` 
- **CLIP**: `qwen_2.5_vl_fp16.safetensors`

### 2. Dependencias Python
El pipeline instalará automáticamente:
```
requests, websocket-client, pillow, base64
```

## Instalación

1. **Copiar el archivo del pipeline**:
   ```bash
   cp comfyui_image_generator_pipeline.py ./pipelines/
   ```

2. **Reiniciar Pipelines**:
   ```bash
   # Si usas Docker
   docker restart pipelines
   
   # Si ejecutas localmente
   curl -X POST "http://localhost:9099/pipelines/reload" \
        -H "Authorization: Bearer 0p3n-w3bu!"
   ```

## Configuración

### Valves Principales

| Valve | Descripción | Valor por Defecto |
|-------|-------------|-------------------|
| `COMFYUI_HOST` | IP del servidor ComfyUI | `192.168.7.101` |
| `COMFYUI_PORT` | Puerto del servidor ComfyUI | `8188` |
| `WIDTH` | Ancho de la imagen | `1024` |
| `HEIGHT` | Alto de la imagen | `1024` |
| `STEPS` | Pasos de difusión | `20` |
| `CFG_SCALE` | Escala CFG | `5.0` |
| `ENHANCE_PROMPTS` | Activar mejora de prompts | `true` |
| `PROMPT_ENHANCEMENT_STYLE` | Estilo de mejora | `detailed` |

### Estilos de Mejora de Prompts

- **`detailed`**: Agrega detalles técnicos y calidad profesional
- **`artistic`**: Estilo artístico y creativo
- **`photorealistic`**: Fotorrealismo y calidad profesional
- **`fantasy`**: Arte fantástico y atmosfera mágica

## Uso

### Comandos de Ejemplo

```
Genera una imagen de un gato con corona
Crea una imagen de un paisaje futurista
Dibuja un robot en un bosque encantado
```

### Palabras Clave Reconocidas

El pipeline detecta automáticamente solicitudes de imagen con estas palabras:
- `genera`, `crea`, `dibuja`, `imagen`
- `picture`, `image`, `draw`, `create`, `generate`

## Flujo de Trabajo

1. **Detección**: El pipeline identifica solicitudes de imagen
2. **Mejora de Prompt**: Opcionalmente mejora el prompt con el estilo seleccionado
3. **Preparación**: Configura el workflow de ComfyUI con parámetros actuales
4. **Envío**: Envía la solicitud a la cola de ComfyUI
5. **Monitoreo**: Sigue el progreso via WebSocket o polling
6. **Descarga**: Obtiene la imagen generada
7. **Visualización**: Muestra la imagen en la conversación

## Actualizaciones de Estado

Durante la generación verás actualizaciones como:

```
🚀 Iniciando generación de imagen...
📝 Prompt: un gato con corona
⚙️ Preparando workflow de ComfyUI...
📤 Enviando solicitud a ComfyUI...
✅ Solicitud enviada (ID: 12345678...)
🔄 Monitoreando progreso de generación...
🎨 Generando imagen... 25% (5/20 pasos)
🎨 Generando imagen... 50% (10/20 pasos)
🎨 Generando imagen... 100% (20/20 pasos)
📥 Descargando imagen generada...
✅ ¡Imagen generada exitosamente!
```

## Solución de Problemas

### Error de Conexión
```
❌ Error al enviar prompt a ComfyUI: Connection refused
```
**Solución**: Verificar que ComfyUI esté ejecutándose en la IP y puerto configurados.

### Modelos No Encontrados
```
❌ Error: Model not found
```
**Solución**: Verificar que los modelos especificados en las valves existan en ComfyUI.

### Timeout
```
❌ Timeout esperando generación de imagen
```
**Solución**: Aumentar el timeout o verificar que ComfyUI no esté sobrecargado.

## Estructura del Template

El pipeline usa el template `templates/comfyui/image_omnigen2_t2i.json` que define:

- **Nodo 6**: Codificación de prompt positivo
- **Nodo 7**: Codificación de prompt negativo  
- **Nodo 11**: Configuración de latents (dimensiones)
- **Nodo 21**: Generación de semilla aleatoria
- **Nodo 23**: Configuración del scheduler (pasos)
- **Nodo 27**: Configuración CFG
- **Nodo 9**: Guardado de imagen

## API ComfyUI Utilizada

- `POST /prompt` - Enviar workflow a la cola
- `WS /ws` - Monitoreo en tiempo real
- `GET /view` - Descargar imagen generada
- `GET /history/{prompt_id}` - Estado de la solicitud

## Desarrollo

Para modificar el pipeline:

1. Editar `comfyui_image_generator_pipeline.py`
2. Actualizar template en `templates/comfyui/` si es necesario
3. Recargar con: `POST /pipelines/reload`

## Licencia

MIT License - Ver archivo principal para detalles completos.
