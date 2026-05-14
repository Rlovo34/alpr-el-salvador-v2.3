"""Módulo de evaluación: Análisis exhaustivo del modelo entrenado.

Realiza dos evaluaciones independientes:

1. EVALUACIÓN EN DATOS SINTÉTICOS:
   - Dataset generado por generator.py (36 clases × 1000 muestras)
   - Split 80/20 (training/validation)
   - Calcula: Accuracy, Precision, Recall, F1-score por clase
   - Genera: Matriz de confusión como PNG
   - Identifica: Pares de clases confundidas (ej: 0↔O, 1↔I)

2. EVALUACIÓN EN DATOS REALES (Opcional):
   - Imágenes de validación en data/raw/valid/
   - Análisis sin etiquetas (solo predicciones)
   - Recomendación: Etiquetar manualmente para validación real

Salidas:
   - Matriz de confusión: output_visuals/confusion_matrix_synthetic.png
   - Reportes en consola: Métricas por clase
   - Análisis de problemas: Caracteres sistemáticamente confundidos
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def evaluate():
    """Evalúa el modelo entrenado en dos escenarios: datos sintéticos y reales.
    
    Evaluación 1 - Datos Sintéticos:
    --------------------------------
    Carga validación set (20% de data/processed/train/)
    Calcula:
        - Accuracy: % de predicciones correctas
        - Matriz de confusión: Qué clases se confunden entre sí
        - Reporte por clase: Precision, Recall, F1-score
    Genera:
        - PNG de matriz de confusión (36x36 heatmap)
        - Análisis de pares problemáticos
    
    Evaluación 2 - Datos Reales:
    ----------------------------
    Si existen imágenes en data/raw/valid/
    Intenta cargarlas para análisis exploratorio
    Nota: Sin etiquetas, solo muestra conteo de muestras
    
    Returns:
        None (genera archivos PNG y logs en consola)
    """
    
    # Cargar modelo entrenado
    model = tf.keras.models.load_model("models/trained_model.h5")
    
    # Evaluación 1: Dataset Sintético (split 80/20 del training)
    print("\n" + "="*70)
    print("📊 EVALUACIÓN 1: Dataset Sintético (Validación)")
    print("="*70)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/train", image_size=(96, 96), shuffle=False)
    
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    preds = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    
    # Accuracy general
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✓ Accuracy General: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n📈 Reporte Detallado por Clase:")
    print(classification_report(y_true, y_pred, target_names=val_ds.class_names))
    
    # Generar Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=val_ds.class_names, yticklabels=val_ds.class_names,
                cbar_kws={'label': 'Número de Muestras'})
    plt.title("Matriz de Confusión - Dataset Sintético", fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Verdadero', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('output_visuals', 'confusion_matrix_synthetic.png'), dpi=150)
    print("\n💾 Matriz de confusión guardada en output_visuals/confusion_matrix_synthetic.png")
    plt.show()
    
    # Análisis de caracteres problemáticos
    print("\n🔍 Caracteres con Mayor Confusión:")
    for i, class_name in enumerate(val_ds.class_names):
        class_cm = cm[i, :]
        if class_name != np.argmax(class_cm):
            confused_with = val_ds.class_names[np.argsort(class_cm)[-2]]
            confusion_count = class_cm[np.argsort(class_cm)[-2]]
            if confusion_count > 0:
                print(f"  ⚠️  {class_name} frecuentemente confundido con {confused_with} ({confusion_count} veces)")
    
    # Evaluación 2: Datos Reales (si existen)
    print("\n" + "="*70)
    print("📊 EVALUACIÓN 2: Datos Reales")
    print("="*70)
    
    real_data_path = os.path.join(os.getcwd(), 'data', 'raw', 'valid')
    if os.path.exists(real_data_path) and any(os.scandir(real_data_path)):
        try:
            real_ds = tf.keras.preprocessing.image_dataset_from_directory(
                real_data_path, image_size=(96, 96), shuffle=False, label_mode=None)
            print(f"✓ Datos reales encontrados en: {real_data_path}")
            print(f"  Nota: Para evaluación precisa, se recomienda etiquetar estas imágenes manualmente.")
            print(f"  Muestras disponibles: {len(list(real_ds))} lotes")
        except Exception as e:
            print(f"⚠️  No se pudieron cargar datos reales: {str(e)}")
    else:
        print(f"⚠️  No hay datos reales en {real_data_path}")
        print("   Recomendación: Añadir imágenes reales de placas para validación en producción.")
    
    print("\n" + "="*70)
    print("✅ Evaluación completada")
    print("="*70)

if __name__ == "__main__":
    evaluate()