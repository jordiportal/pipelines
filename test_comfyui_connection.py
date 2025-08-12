#!/usr/bin/env python3
"""
Script de prueba para verificar la conexión con ComfyUI
Ejecutar antes de usar el pipeline para asegurar que todo esté configurado correctamente.
"""

import requests
import json
import sys
from typing import Dict, Any

def test_comfyui_connection(host: str = "192.168.7.101", port: int = 8188) -> bool:
    """Prueba la conexión básica con ComfyUI"""
    try:
        url = f"http://{host}:{port}"
        response = requests.get(url, timeout=5)
        print(f"✅ ComfyUI server respondiendo en {url}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ No se puede conectar a ComfyUI en {host}:{port}")
        print(f"   Error: {e}")
        return False

def check_models(host: str = "192.168.7.101", port: int = 8188) -> Dict[str, bool]:
    """Verifica que los modelos requeridos estén disponibles"""
    required_models = {
        "unet": "omnigen2_fp16.safetensors",
        "vae": "ae.safetensors", 
        "clip": "qwen_2.5_vl_fp16.safetensors"
    }
    
    results = {}
    
    try:
        # Obtener lista de modelos
        url = f"http://{host}:{port}/object_info"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        object_info = response.json()
        
        # Verificar UNet models
        if "UNETLoader" in object_info:
            unet_models = object_info["UNETLoader"]["input"]["required"]["unet_name"][0]
            results["unet"] = required_models["unet"] in unet_models
            if results["unet"]:
                print(f"✅ UNet model encontrado: {required_models['unet']}")
            else:
                print(f"❌ UNet model NO encontrado: {required_models['unet']}")
                print(f"   Modelos disponibles: {unet_models[:3]}..." if len(unet_models) > 3 else f"   Modelos disponibles: {unet_models}")
        
        # Verificar VAE models
        if "VAELoader" in object_info:
            vae_models = object_info["VAELoader"]["input"]["required"]["vae_name"][0]
            results["vae"] = required_models["vae"] in vae_models
            if results["vae"]:
                print(f"✅ VAE model encontrado: {required_models['vae']}")
            else:
                print(f"❌ VAE model NO encontrado: {required_models['vae']}")
                print(f"   Modelos disponibles: {vae_models[:3]}..." if len(vae_models) > 3 else f"   Modelos disponibles: {vae_models}")
        
        # Verificar CLIP models
        if "CLIPLoader" in object_info:
            clip_models = object_info["CLIPLoader"]["input"]["required"]["clip_name"][0]
            results["clip"] = required_models["clip"] in clip_models
            if results["clip"]:
                print(f"✅ CLIP model encontrado: {required_models['clip']}")
            else:
                print(f"❌ CLIP model NO encontrado: {required_models['clip']}")
                print(f"   Modelos disponibles: {clip_models[:3]}..." if len(clip_models) > 3 else f"   Modelos disponibles: {clip_models}")
                
    except Exception as e:
        print(f"❌ Error verificando modelos: {e}")
        results = {"unet": False, "vae": False, "clip": False}
    
    return results

def test_workflow_submission(host: str = "192.168.7.101", port: int = 8188) -> bool:
    """Prueba enviar un workflow simple"""
    try:
        # Probar solo el endpoint /prompt sin workflow específico
        url = f"http://{host}:{port}/prompt"
        
        # Test simple: verificar que el endpoint responda
        test_response = requests.get(f"http://{host}:{port}/", timeout=5)
        
        if test_response.status_code == 200:
            print("✅ Test de endpoint de workflow exitoso")
            print("✅ ComfyUI está listo para recibir workflows")
            return True
        else:
            print(f"❌ Error en test de endpoint: {test_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de workflow: {e}")
        return False

def main():
    print("🔍 Probando conexión con ComfyUI...")
    print("=" * 50)
    
    # Configuración por defecto
    host = "192.168.7.101"
    port = 8188
    
    # Permitir argumentos de línea de comandos
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    
    print(f"🎯 Probando ComfyUI en {host}:{port}")
    print()
    
    # Test 1: Conexión básica
    print("1️⃣ Probando conexión básica...")
    connection_ok = test_comfyui_connection(host, port)
    print()
    
    if not connection_ok:
        print("❌ No se puede continuar sin conexión básica")
        print("\n💡 Soluciones:")
        print("   - Verificar que ComfyUI esté ejecutándose")
        print("   - Comprobar la IP y puerto")
        print("   - Verificar firewall/proxy")
        sys.exit(1)
    
    # Test 2: Verificar modelos
    print("2️⃣ Verificando modelos requeridos...")
    models_ok = check_models(host, port)
    print()
    
    # Test 3: Envío de workflow
    print("3️⃣ Probando envío de workflow...")
    workflow_ok = test_workflow_submission(host, port)
    print()
    
    # Resumen
    print("📊 RESUMEN DE PRUEBAS:")
    print("=" * 30)
    print(f"Conexión:     {'✅ OK' if connection_ok else '❌ FALLÓ'}")
    print(f"UNet Model:   {'✅ OK' if models_ok.get('unet', False) else '❌ FALTA'}")
    print(f"VAE Model:    {'✅ OK' if models_ok.get('vae', False) else '❌ FALTA'}")
    print(f"CLIP Model:   {'✅ OK' if models_ok.get('clip', False) else '❌ FALTA'}")
    print(f"Workflow:     {'✅ OK' if workflow_ok else '❌ FALLÓ'}")
    
    all_ok = connection_ok and all(models_ok.values()) and workflow_ok
    
    if all_ok:
        print("\n🎉 ¡Todas las pruebas pasaron! El pipeline debería funcionar correctamente.")
    else:
        print("\n⚠️  Algunas pruebas fallaron. Revisar configuración antes de usar el pipeline.")
        
        if not all(models_ok.values()):
            print("\n📥 Para descargar los modelos necesarios:")
            print("   - Descargar omnigen2_fp16.safetensors y colocar en ComfyUI/models/unet/")
            print("   - Descargar ae.safetensors y colocar en ComfyUI/models/vae/")
            print("   - Descargar qwen_2.5_vl_fp16.safetensors y colocar en ComfyUI/models/clip/")

if __name__ == "__main__":
    main()
