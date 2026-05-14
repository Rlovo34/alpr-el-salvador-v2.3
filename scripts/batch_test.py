import os
import sys
import glob

# 1. Configuración de rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(os.path.join(project_root, "src"))

from inference.predict import PlatePredictor

def evaluate_bulk():
    # Apuntamos a la base de la carpeta raw
    raw_dir = os.path.join(project_root, "data", "raw")
    model_path = os.path.join(project_root, "models", "trained_model.h5")
    
    # 2. Búsqueda RECURSIVA: Busca en todas las subcarpetas (valid, alabama, etc.)
    # El patrón **/ *.jpg busca archivos en cualquier nivel de profundidad
    search_pattern = os.path.join(raw_dir, "**", "*.jpg")
    image_paths = glob.glob(search_pattern, recursive=True)
    
    if not image_paths:
        print(f"❌ Error: No se encontraron imágenes en {raw_dir}")
        print("Asegúrate de que las carpetas /valid/alabama/... estén dentro de data/raw/")
        return
        
    print(f"--- 🚀 Iniciando Evaluación Masiva (MEJORADO v2.2) ---")
    print(f"Imágenes detectadas en subcarpetas: {len(image_paths)}")
    
    try:
        predictor = PlatePredictor(model_path)
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    print(f"\n{'ESTADO':<18} | {'PREDICCIÓN':<12} | {'CONF':<6} | {'ARCHIVO':<30}")
    print("-" * 75)
    
    stats = {'total': 0, 'low_confidence': 0, 'confidence_sum': 0.0}
    
    # Procesamos un máximo de 100 imágenes
    for img_path in image_paths[:100]:
        # Extraemos el nombre del estado (carpeta padre) y el nombre del archivo
        partes = img_path.split(os.sep)
        estado = partes[-2] # La carpeta justo arriba de la imagen
        filename = partes[-1]
        
        try:
            result = predictor.run(img_path)
            texto = result['text'] if result['text'] else "[VACÍO]"
            confianza = result['confidence']
            
            # Estadísticas
            stats['total'] += 1
            stats['confidence_sum'] += confianza
            
            if confianza < 0.55:
                stats['low_confidence'] += 1
                
            conf_str = f"{confianza*100:.1f}%"
            print(f"{estado[:18]:<18} | {texto:<12} | {conf_str:<6} | {filename[:30]:<30}")
        except Exception as e:
            stats['total'] += 1
            print(f"{estado[:18]:<18} | {'[ERROR]':<12} | {'N/A':<6} | {filename[:30]:<30}")
            
    print("-" * 75)
    
    # Estadísticas finales
    avg_confidence = stats['confidence_sum'] / max(1, stats['total']) if stats['total'] > 0 else 0
    
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"  • Total procesadas: {stats['total']}")
    print(f"  • Confianza promedio: {avg_confidence*100:.1f}%")
    print(f"  • Confianza baja (<55%): {stats['low_confidence']} ({100*stats['low_confidence']//max(1,stats['total'])}%)")
    print(f"\n💡 INTERPRETAR RESULTADOS:")
    print(f"  • CONF Alta (>70%): Predicción muy confiable")
    print(f"  • CONF Media (55-70%): Predicción aceptable")
    print(f"  • CONF Baja (<55%): Caracteres rechazados por filtro")
    print(f"\n✅ Prueba completada. Revisa predicciones con confianza baja para mejorar segmentación.")
    

if __name__ == "__main__":
    evaluate_bulk()