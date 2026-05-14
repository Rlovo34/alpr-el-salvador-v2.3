import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de rutas robusta
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Importamos tu motor de inferencia
from src.inference.predict import PlatePredictor
from tensorflow.keras.applications.resnet50 import preprocess_input

def visualize_sv_pipeline():
    # Definición de rutas según tu estructura
    model_path = os.path.join(project_root, 'models', 'trained_model.h5')
    # RUTA EXACTA:  data\raw\valid\1_EL _SALVADOR
    valid_dir = os.path.join(project_root, 'data', 'raw', 'valid', '1_EL _SALVADOR')
    output_dir = os.path.join(project_root, 'output_visuals')
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(valid_dir):
        print(f"❌ ERROR: No se encuentra la carpeta en: {valid_dir}")
        return

    print("🚀 Iniciando Visualizador de Pipeline v2.2 (El Salvador)...")
    predictor = PlatePredictor(model_path)

    # Filtrar solo archivos de imagen
    image_files = [f for f in os.listdir(valid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(valid_dir, img_name)
        
        # --- PROCESAMIENTO UNIFICADO (usa run() que incluye el filtro de confianza) ---
        # Paso 1-3: Preprocesamiento → Segmentación → Clasificación
        result = predictor.run(img_path, min_char_confidence=0.55)
        
        final_text = result['text']
        confidence = result['confidence']
        min_confidence = result['min_confidence']
        weak_chars = result['weak_chars']  # Índices de caracteres débiles
        
        # --- OBTENER DETALLES INTERNOS PARA VISUALIZACIÓN ---
        thresh, img_std = predictor.preprocess_image(img_path)
        char_images = predictor.extract_characters(thresh, img_std)
        
        # Detalle: obtener predicciones por carácter
        char_preds = []
        char_confs = []
        if char_images:
            batch = np.array(char_images, dtype=np.float32)
            batch = preprocess_input(batch)
            predictions = predictor.model.predict(batch, verbose=0)
            for pred in predictions:
                char = predictor.classes[np.argmax(pred)]
                conf = np.max(pred)
                char_preds.append(char)
                char_confs.append(conf)

        # --- GENERACIÓN DEL DASHBOARD ---
        num_chars = len(char_images)
        
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(
            f"Pipeline OCR v2.2 | {img_name}\nResultado: {final_text} | Confianza: {confidence*100:.1f}%",
            fontsize=16, fontweight='bold'
        )

        # 1. Imagen Original Normalizada
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB))
        ax1.set_title("1. Imagen Estandarizada\n(600px ancho)", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # 2. Máscara Black-Hat + Otsu (La "magia" para El Salvador)
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(thresh, cmap='gray')
        ax2.set_title("2. Black-Hat + Otsu\n(Extracción de caracteres)", fontsize=11, fontweight='bold')
        ax2.axis('off')

        # 3. Panel de Análisis con Datos
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        
        # Construir tabla de caracteres identificados
        char_table = "CARACTERES IDENTIFICADOS\n" + "="*40 + "\n"
        if char_preds:
            for i, (char, conf) in enumerate(zip(char_preds, char_confs)):
                confidence_bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                # Marcar caracteres débiles con asterisco
                weak_mark = " ⚠️ DÉBIL" if i in weak_chars else ""
                char_table += f"{i+1}. '{char}' {conf*100:5.1f}% [{confidence_bar}]{weak_mark}\n"
        else:
            char_table += "⚠️  No se detectaron caracteres\n"
        
        char_table += "\n" + "="*40 + "\n"
        char_table += f"ESTADÍSTICAS\n" + "="*40 + "\n"
        char_table += f"• Total caracteres: {num_chars}\n"
        char_table += f"• Caracteres débiles: {len(weak_chars)}\n"
        char_table += f"• Confianza promedio: {confidence*100:.1f}%\n"
        char_table += f"• Confianza mínima: {min_confidence*100:.1f}%\n"
        char_table += f"• Confianza máxima: {max(char_confs)*100:.1f}%" if char_confs else "• Confianza máxima: N/A\n"
        char_table += f"\n" + "="*40 + "\n"
        char_table += f"PROCESAMIENTO v2.3\n" + "="*40 + "\n"
        char_table += f"• Incluir débiles: SÍ\n"
        char_table += f"• Umbral débil: <55%\n"
        char_table += f"• Área mínima: 0.15%\n"
        char_table += f"• Aspect ratio: 0.12-1.15\n"
        
        ax3.text(0.05, 0.95, char_table, transform=ax3.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # 4-11. Caracteres Segmentados y Reconocidos
        if num_chars > 0:
            # Limitar a máximo 8 caracteres para visualización (caben en 8 espacios)
            for i in range(min(num_chars, 8)):
                ax_char = plt.subplot(2, 8, 8 + i + 1)
                ax_char.imshow(char_images[i])
                
                # Color del título según confianza
                char_conf = char_confs[i] if i < len(char_confs) else 0
                color = 'green' if char_conf >= 0.7 else 'orange' if char_conf >= 0.55 else 'red'
                
                # Indicar si es carácter débil
                weak_indicator = " ⚠️" if i in weak_chars else ""
                
                ax_char.set_title(
                    f"{char_preds[i]}{weak_indicator}\n{char_conf*100:.0f}%",
                    fontsize=11, fontweight='bold', color=color
                )
                
                # Añadir borde rojo para caracteres débiles
                if i in weak_chars:
                    for spine in ax_char.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                        spine.set_visible(True)
                else:
                    for spine in ax_char.spines.values():
                        spine.set_visible(False)
                
                ax_char.axis('off')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Guardar el resultado
        save_path = os.path.join(output_dir, f"visual_{img_name}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Dashboard v2.3 guardado: {os.path.basename(save_path)}")
        print(f"   └─ Predicción: {final_text} | Confianza: {confidence*100:.1f}% | Caracteres débiles: {len(weak_chars)}\n")
        
        # Mostrar en pantalla
        plt.show()

if __name__ == "__main__":
    visualize_sv_pipeline()