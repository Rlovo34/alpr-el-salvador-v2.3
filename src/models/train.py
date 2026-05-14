"""Módulo de entrenamiento: Fine-tuning de ResNet50 para OCR de placas SV.

Este módulo implementa un entrenamiento en DOS FASES de una red ResNet50:

FASE 1 - WARM-UP (10 épocas):
    - Red base congelada (ImageNet como extractor de features)
    - Solo entrena capas clasificadoras superiores
    - Learning rate alto (0.001) para convergencia rápida
    - Objetivo: Adaptar características ImageNet a nuestro dataset
    
FASE 2 - FINE-TUNING (15 épocas):
    - Descongelamiento parcial: últimas 30 capas de ResNet50
    - Learning rate bajísimo (0.00005) para no destruir features ImageNet
    - Callbacks: ReduceLROnPlateau, EarlyStopping
    - Objetivo: Refinamiento fino de todas las capas

Augmentación de datos:
    - Aplicada dinámicamente en cada época (ImageDataGenerator)
    - Complementa el dataset sintético pre-generado
    - Simula variabilidad real: ángulos, iluminación, zoom

Arquitectura final:
    ResNet50(pre-trained) → GlobalAveragePooling → Dense(512) → 
    BatchNormalization → Dropout(0.5) → Dense(36, softmax)
    
Parámetros clave justificados:
    - Dense(512): Ancho suficiente para 36 clases sin sobreajuste
    - Dropout(0.5): Reduce co-adaptación de neuronas
    - BatchNormalization: Estabiliza gradientes, mejor convergencia
    - Últimas 30 capas: ResNet50 tiene ~170 capas, descongelar solo las últimas
                       evita overfitting manteniendo features genéricas
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIR_DATOS = os.path.join(BASE_DIR, 'data', 'processed', 'train')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.h5')

# --- CONFIGURACIÓN DE AUGMENTACIÓN ---
# Aplicada dinámicamente DURANTE el entrenamiento para aumentar variabilidad
# Complementa el dataset sintético pre-generado (generator.py)
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,           # ±15°: simula ángulos de captura inclinados
    width_shift_range=0.15,      # ±15%: desplazamiento horizontal (placas mal centradas)
    height_shift_range=0.15,     # ±15%: desplazamiento vertical
    shear_range=0.20,            # ±20%: deformación angular (perspectiva)
    zoom_range=0.20,             # 0.8x - 1.2x: simula distancia variable a cámara
    brightness_range=[0.7, 1.3], # 70%-130%: variación de iluminación/reflexión
    fill_mode='constant',        # Relleno con negro (color de fondo típico)
    cval=0,
    validation_split=0.2         # Split 80% train / 20% validation
)

# Validación SIN augmentación (para reproducibilidad de métricas)
datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_gen = datagen_train.flow_from_directory(
    DIR_DATOS, 
    target_size=(96, 96), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training'
)

val_gen = datagen_val.flow_from_directory(
    DIR_DATOS, 
    target_size=(96, 96), 
    batch_size=32, 
    class_mode='categorical', 
    subset='validation', 
    shuffle=False
)

# --- ARQUITECTURA DEL MODELO: TRANSFER LEARNING ---
# Cargamos ResNet50 pre-entrenada en ImageNet (pesos de 151 capas)
# include_top=False: Descartamos capas clasificadoras de ImageNet
# Esto permite usar el "backbone" de extracción de features para nuestro problema
base_model = ResNet50(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# 🔥 FASE 1: WARM-UP (Congelamiento Total de la Red Base)
# =====================================================
# Objetivo: Adaptar las capas de clasificación a nuestro dataset 
#           sin dañar los features aprendidos en ImageNet
# Duración: 10 épocas (suficiente para convergencia de capas simples)
# Learning rate: Alto (0.001) porque partimos de pesos aleatorios en nuevas capas
base_model.trainable = False 

# Cabeza clasificadora personalizada para 36 clases
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce cada feature map a escalar (es más robusto que Flatten)
x = Dense(512, activation='relu')(x)  # 512 neuronas: balance entre capacidad y regularización
x = BatchNormalization()(x)  # Normaliza entre capas: acelera convergencia, mejora generalización
x = Dropout(0.5)(x)  # Desactiva 50% de neuronas: reduce overfitting
predictions = Dense(train_gen.num_classes, activation='softmax')(x)  # 36 clases

model = Model(inputs=base_model.input, outputs=predictions)

print("\n" + "="*70)
print("🚀 FASE 1: Calentamiento de Capa Clasificadora (Red Base Congelada)")
print("="*70)
print(f"Clases detectadas: {train_gen.num_classes}")
print(f"Muestras de entrenamiento: {train_gen.n}")
print(f"Muestras de validación: {val_gen.n}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # LR alto para convergencia rápida
    loss='categorical_crossentropy',  # Estándar para clasificación multiclase
    metrics=['accuracy']
)

# Entrenamiento Fase 1
model.fit(
    train_gen, 
    epochs=10,  # 10 épocas: normalmente suficiente para capas nuevas
    validation_data=val_gen
)

# 🔥 FASE 2: FINE-TUNING (Descongelamiento Parcial)
# ==================================================
# Objetivo: Refinar pesos de ResNet50 para especializarlos en OCR de placas
# Estrategia: Solo descongelamos las últimas 30 capas (ResNet tiene ~170)
#            Las primeras capas extraen features genéricas (bordes, texturas)
#            Las últimas capas extraen features específicas del dominio
# Duración: hasta 15 épocas (con EarlyStopping si no mejora)
# Learning rate: BAJÍSIMO (0.00005) para fine-tuning sin destruir features pre-entrenadas

print("\n" + "="*70)
print("🧪 FASE 2: Fine-Tuning de ResNet50 (Últimas 30 Capas)")
print("="*70)

base_model.trainable = True

# Mantener congeladas las primeras 140 capas (de 170)
# Solo entrenar las últimas 30 capas (aproximadamente el último bloque residual)
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # LR muy bajo: protege features ImageNet
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks para entrenamiento inteligente
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # Monitoreamos pérdida en validación
    factor=0.5,              # Reducir LR en 50% si no mejora
    patience=2,              # Después de 2 épocas sin mejora
    min_lr=0.000001,         # Nunca bajar de 1e-6
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',  # Monitoreamos exactitud en validación
    patience=5,              # Detener si no mejora en 5 épocas
    restore_best_weights=True,  # Restaurar pesos de mejor época
    verbose=1
)

# Entrenamiento Fase 2
model.fit(
    train_gen, 
    epochs=15,  # Máximo 15 épocas (EarlyStopping puede detener antes)
    validation_data=val_gen, 
    callbacks=[reduce_lr, early_stop]
)

# Guardar modelo final
model.save(MODEL_PATH)
print(f"\n✅ Modelo ResNet50 entrenado y guardado en {MODEL_PATH}")