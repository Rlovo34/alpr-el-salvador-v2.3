# OCR Híbrido para Placas de El Salvador

## 📋 Descripción del Proyecto

Sistema integral de reconocimiento óptico de caracteres (OCR) especializado en la detección y lectura automática de placas vehiculares salvadoreñas. Implementa una arquitectura híbrida que combina **visión por computadora avanzada (OpenCV)** con **aprendizaje profundo (ResNet50)** para lograr resultados robustos bajo diversas condiciones de iluminación y ángulos de captura.

**Características principales:**
- Preprocesamiento inteligente mediante filtros morfológicos Black-Hat y umbralización de Otsu
- Red neuronal convolucional ResNet50 transferida y refinada
- Segmentación de caracteres adaptativos
- Clasificación de 36 clases (dígitos 0-9 y letras A-Z)
- Soporte para dataset sintético binario y datos reales de validación

---

## 🏗️ Arquitectura del Sistema

El sistema opera en tres fases integradas:

### **Fase 1: Preprocesamiento (OpenCV)**
- **Estandarización**: Redimensiona la imagen a 600px de ancho manteniendo relación de aspecto
- **Conversión a escala de grises**: Optimiza el procesamiento
- **Filtro Black-Hat**: Extrae caracteres oscuros sobre fondo claro (crítico para placas reflectantes)
- **Umbralización de Otsu**: Convierte a imagen binaria automáticamente
- **Dilatación**: Suelda trazos rotos causados por reflejos del metal

### **Fase 2: Segmentación de Caracteres**
- Detecta contornos individuales en la imagen binarizada
- Extrae cada carácter como región de interés (ROI)
- Aplica padding dinámico (18% vertical, 40% horizontal)
- Redimensiona cada carácter a 96×96 píxeles normalizados

### **Fase 3: Clasificación con ResNet50 (Aprendizaje Profundo)**

#### **Etapa de Entrenamiento: Warm-Up (10 épocas)**
- Red base ResNet50 congelada
- Entrenamiento exclusivo de capas clasificadoras
- Dense(512) + Dropout(0.5) → Dense(36, softmax)
- Optimizador Adam con learning_rate=0.001

#### **Etapa de Fine-Tuning (N épocas)**
- Descongelamiento de últimas 30 capas de ResNet50
- Aprendizaje diferenciado con learning_rate=0.00001
- Callbacks: ReduceLROnPlateau, EarlyStopping

---

## 🆕 Novedades en v2.3

### **Inclusión de Caracteres Débiles (Breaking Change)**
A diferencia de v2.2, v2.3 **incluye TODOS los caracteres** en el resultado final, incluso si algunos tienen baja confianza (<55%). Esto permite obtener placas completas:

**Antes v2.2:**
```
Placa esperada: P18183 (5 caracteres)
Resultado: P1818 (4 caracteres - descartó por confianza 32%)
```

**Ahora v2.3:**
```
Placa esperada: P18183 (5 caracteres)
Resultado: P18183 (5 caracteres completos)
Panel: Carácter 4 ⚠️ DÉBIL (32%) [borde rojo]
```

**Cambios Técnicos v2.3:**
- ✅ Método `run()` incluye TODOS los caracteres sin filtro
- ✅ Nuevo campo: `weak_chars` (índices de caracteres <55%)
- ✅ Visualización marca caracteres débiles con **⚠️ DÉBIL** y **borde rojo**
- ✅ Panel de análisis muestra cantidad de caracteres débiles
- ✅ Parámetro `min_char_confidence` ahora solo informativo (no filtra)

---

## 🆕 Mejoras en v2.2

### **Segmentación Más Restrictiva**
Cambios para reducir caracteres fantasma:

**Filtros Mejorados:**
- **Área mínima:** 0.0005 → **0.0015** (3x más restrictivo)
- **Aspect ratio:** 0.10-1.30 → **0.12-1.15** (rechaza filamentos/manchas)
- **Altura:** 20-80% → **25-75%** (menos ruido en bordes)
- **Banda vertical:** 25-75% → **30-70%** (centrado más estricto)
- **Erosión pre-contornos:** Remueve artefactos microscópicos

**Preprocesamiento Mejorado:**
- **Limpieza morfológica:** Apertura (erosión+dilatación) + Cierre
- **Efecto:** Elimina reflexiones y artefactos sin dañar caracteres reales
- **Parámetros:** kernel adaptativo (3.8% del ancho de imagen)

**Validación Flexible:**
- **Antes:** Exactamente 8 caracteres
- **Ahora:** 6-8 caracteres (tolera descartes por baja confianza)
- **Formato aceptado:**
  - Estricta: ABC12345 (2 letras + 5 números)
  - Flexible: Mixta con ≥2 letras y ≥3 números iniciando con letra

**Scripts Mejorados:**
- ✅ `batch_test.py` con estadísticas detalladas
- ✅ `visualize_sv.py` con panel de análisis integrado
- ✅ `run_inference.py` alineado con nuevo formato v2.2


## 🆕 Novedades en v2.1

### **Mejoras en `run_inference.py`**
Hasta v2.0, el script solo mostraba resultados en consola. Ahora genera automáticamente dashboards visuales:

**Cambios Principales:**
- ✨ **Generación automática de visualizaciones** idénticas a `visualize_sv.py`
- ✨ **Parámetro `--no-visual`** para deshabilitar visualización si es necesario
- ✨ **Mejor gestión de rutas** (soporta rutas relativas y absolutas)
- ✨ **Creación automática de carpeta** `output_visuals/` si no existe
- ✨ **Función `generate_visualization()`** reutilizable para otros scripts

**Ejemplo de Uso:**
```bash
# Con visualización (por defecto)
python scripts/run_inference.py --input "data/raw/valid/1_EL _SALVADOR/placa.jpg"

# Solo texto (sin generar PNG)
python scripts/run_inference.py --input "datos/placa.jpg" --no-visual
```

### **Mejoras en Documentación Interna**
Se realizó refactorización completa de comentarios y Docstrings:

**Archivos Actualizados:**
1. **`src/inference/predict.py`** - Docstrings profesionales (formato Google)
   - Documentación de clase `PlatePredictor`
   - Explicación del "por qué matemático" (kernel 3.8%, padding 18%/40%)
   - Detalles de cada fase: preprocesamiento, segmentación, clasificación
   
2. **`src/models/train.py`** - Justificación de hiperparámetros
   - Explicación de por qué 10 épocas en Warm-Up
   - Justificación de descongelar últimas 30 capas
   - Learning rates diferenciados (0.001 vs 0.00005)
   
3. **`src/models/evaluate.py`** - Documentación de evaluación
   - Cálculo de métricas por clase
   - Análisis de matriz de confusión
   - Manejo de datos reales vs sintéticos
   
4. **`src/data/generator.py`** - Documentación de augmentación
   - Justificación de 1000 muestras/clase
   - Transformaciones: rotación, shear, zoom, ruido morfológico
   - Fondos variables (negro, gris, reflejos metálicos)

### **Cambios Técnicos (Backend)**
- Eliminado código comentado obsoleto
- Añadidos Docstrings en formato Google en todas las funciones
- Comentarios enfocados en "por qué" vs narrar código evidente
- Parámetros técnicos justificados matemáticamente


## 🆕 Novedades en v2.0 

Basado en análisis de arquitectura, se han implementado **10 mejoras clave** para aumentar la precisión de identificación con respecto a la version 1.0:

### **1. Augmentación de Datos Agresiva** ⭐⭐⭐⭐⭐
- **Dataset aumentado:** 250 → **1000 muestras por clase** (36,000 imágenes totales)
- **Transformaciones aplicadas:**
  - Rotación: ±15° (simula ángulos de captura)
  - Zoom: 0.85x - 1.15x (simula distancias variables)
  - Shear: ±20% (deformación angular)
  - Desplazamiento: ±15% horizontal/vertical
  - Brillo: 0.7x - 1.3x (variación de iluminación)
  - Ruido Gaussiano y morfológico
- **Fondos variables:** Negro puro, gris, reflejos metálicos (no solo negro)

### **2. BatchNormalization en el Clasificador** ⭐⭐⭐
- Añadida capa `BatchNormalization()` post-Dense(512)
- Beneficios:
  - Estabilización del entrenamiento
  - Mejor convergencia
  - Reducción de overfitting
  - Mayor generalización a datos reales

### **3. Preprocesamiento Adaptativo** ⭐⭐
- Kernel del filtro Black-Hat ahora es **dinámico** (basado en %ancho de imagen)
- Fórmula: `kernel_size = int(img_width * 0.038)` (3.8% del ancho)
- Beneficios: Mejor adaptación a imágenes de diferentes resoluciones

### **4. Segmentación de Caracteres Relajada** ⭐⭐⭐
- **Filtros menos restrictivos:**
  - Altura: 20-80% (antes 30-75%) → Captura caracteres más pequeños
  - Aspect ratio: 0.10-1.30 (antes 0.15-0.95) → Permite letras estrechas (I, l)
  - Posición vertical: 25-75% (antes 35-65%) → Mayor flexibilidad
- **Umbral de área dinámico:** 0.05% del área total de imagen
- Resultado: Recupera caracteres que antes se descartaban

### **5. Validación de Patrones de Placas** ⭐⭐⭐
- Método `validate_plate_format()` en `PlatePredictor`
- Verifica dos variantes de formato SV:
  - Estricta: ABC12345 (3 letras + 5 números)
  - Flexible: Cualquier combinación con ≥2 letras y ≥3 números
- Rechaza predicciones imposibles (ej: 8 números, 0 letras)
- Retorna: `valid_format` (bool) en respuesta

### **6. Salidas Detalladas de Predicción** ⭐⭐
- Antes: `{"text": "...", "confidence": 0.XX}`
- Ahora: 
  ```json
  {
    "text": "SV123456",
    "confidence": 0.92,          // Confianza promedio
    "min_confidence": 0.78,      // Carácter menos confiable
    "valid_format": true         // Validación de patrón
  }
  ```
- Permite identificar predicciones débiles

### **7. Augmentación en Entrenamiento (ImageDataGenerator)** ⭐⭐⭐
- Aplicada **dinámicamente** durante cada época
- Independiente del generador sintético
- Amplía variabilidad: la red ve ejemplos nuevos cada vuelta

### **8. Evaluación Mejorada** ⭐⭐
- Matriz de confusión guardada como PNG
- Análisis automático de pares "problemáticos" (ej: 0↔O, 1↔I)
- Soporte para datos reales (cuando están disponibles)

### **9. Fuentes Adicionales** ⭐
- Añadida `FONT_HERSHEY_TRIPLEX` al generador
- Aumenta variabilidad de estilos de caracteres

### **10. Documentación Mejorada** ⭐
- Comentarios detallados en cambios de código
- Docstrings en nuevos métodos
- Logs informativos durante la ejecución

---

---

## 🆕 Información de Versión (Resumen)

| Componente | v2.0 | v2.1 | v2.2 | v2.3 |
|-----------|------|------|------|------|
| **Dataset** | 36,000 img | ✓ | ✓ | ✓ |
| **Modelo** | ResNet50 | ✓ | ✓ | ✓ |
| **Inferencia** | Consola | Visualización | Segmentación mejorada | Incluir débiles |
| **Documentación** | Básica | Profesional | Parámetros técnicos | Caracteres débiles |
| **Scripts** | Básicos | Avanzados | Estadísticas | Panel análisis |
| **Validación** | Patrón simple | Flexible | Flexible | Flexible (6-8 chars) |

---


## 📁 Estructura de Directorios

```
PROYECTO PLACAS/
├── 📄 README.md                    # Este archivo
├── 📄 requirements.txt             # Dependencias del proyecto
│
├── 📂 data/                        # Datasets organizados
│   ├── raw/                        # Datos crudos de validación
│   │   └── valid/
│   │       ├── 1_EL _SALVADOR/     # Imágenes de El Salvador
│   │       ├── ALABAMA/            # Referencias comparativas
│   │       ├── CALIFORNIA/
│   │       └── ... (otros estados)
│   │
│   └── processed/                  # Datos pre-procesados para entrenamiento
│       └── train/
│           ├── 0 - 9/             # Carpetas para dígitos (1000 muestras c/u)
│           └── A - Z/             # Carpetas para letras (1000 muestras c/u)
│
├── 📂 models/                      # Modelos entrenados
│   └── trained_model.h5           # Modelo ResNet50 serializado (Keras)
│
├── 📂 output_visuals/              # Resultados visuales de predicción
│   ├── visual_1.jpg
│   ├── visual_2.jpg
│   └── ... (salidas del pipeline)
│
├── 📂 scripts/                     # Scripts de utilidad
│   ├── batch_test.py              # Evaluación masiva de imágenes
│   ├── run_inference.py           # Inferencia en imagen individual
│   └── visualize_sv.py            # Visualización del pipeline completo
│
└── 📂 src/                         # Módulos reutilizables
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── generator.py            # Generador sintético de dataset
    ├── models/
    │   ├── __init__.py
    │   ├── train.py               # Script de entrenamiento
    │   └── evaluate.py            # Evaluación de métricas
    └── inference/
        ├── __init__.py
        └── predict.py             # Motor de predicción e inferencia
```

---

## 🔧 Requisitos Previos e Instalación

### **Requisitos del Sistema**
- Python 3.8 o superior
- CUDA 11.0+ (opcional, acelera GPU)
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio disponible

### **Pasos de Instalación**

#### 1. **Clonar o descargar el proyecto**
```bash
# Navegar a la carpeta del proyecto
cd "c:\Users\usuario\PROYECTO PLACAS"
```

#### 2. **Descargar MODELO_2.3.rar y agregar las carpetas Faltantes al proyecto ya que por su tamaño no fueron subidas al repositorio**
- `data` - Dataset sintetico y datos de prueba
- `models` - Modelo .h5 preentrenado
- `output_visuals` - salidas de datos de pruebas
```bash
https://drive.google.com/drive/folders/1I3MJ1wtMx_JNavS3nyOr5-qcG5X42L-w?usp=sharing

```
#### 3. **Crear entorno virtual (recomendado)**
```powershell
# Crear entorno virtual
python -m venv parqueo_ocr

# Activar en Windows PowerShell
.\parqueo_ocr\Scripts\Activate.ps1

# Activar en Windows CMD
parqueo_ocr\Scripts\activate.bat

# Activar en macOS/Linux
source parqueo_ocr/bin/activate
```

#### 4. **Instalar dependencias**
```bash
# Instalar paquetes desde requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. **Verificar instalación**
```python
python -c "import tensorflow; import cv2; import numpy; print('✓ Dependencias OK')"
```

### **Dependencias Principales**
- `tensorflow>=2.10.0` - Aprendizaje profundo
- `keras>=2.10.0` - API de modelos
- `opencv-python>=4.6.0` - Visión por computadora
- `numpy>=1.21.0` - Computación numérica
- `matplotlib>=3.5.0` - Visualización

---

## 🚀 Instrucciones de Uso

### **1. Entrenamiento del Modelo (Opcional)**

Si deseas entrenar un nuevo modelo:

```bash
# Primero, generar dataset sintético
python src/data/generator.py

# Luego, entrenar el modelo
python src/models/train.py

# Evaluar rendimiento en validación
python src/models/evaluate.py
```

**Resultado esperado:** Archivo `models/trained_model.h5` actualizado

---

### **2. Inferencia en Imagen Individual**

Procesar una sola imagen de placa con visualización automática del pipeline:

```bash
python scripts/run_inference.py --input "ruta/a/imagen.jpg"
```

**Ejemplo:**
```bash
python scripts/run_inference.py --input "data/raw/valid/1_EL _SALVADOR/placa_001.jpg"
```

**Salida del programa (Consola):**
```
==================================================
Input: data/raw/valid/1_EL _SALVADOR/placa_001.jpg
Prediction: "SV123456"
Confidence: 0.92
Weak Characters: 1 [3]
==================================================

✅ Dashboard v2.3 guardado en: output_visuals/visual_placa_001.jpg
📊 Visualización completada: SV123456 | Confianza: 92.0% | Caracteres débiles: 1
```

**Salida Visual (PNG) v2.3:**
Genera automáticamente un dashboard con 4 paneles principales:
1. **Imagen Estandarizada (600px):** Entrada preprocesada
2. **Máscara Black-Hat + Otsu:** Extracción de caracteres (mejorada v2.2)
3. **Panel de Análisis:** 
   - Lista de caracteres con barras de confianza
   - Caracteres débiles marcados con **⚠️ DÉBIL**
   - Estadísticas (total, débiles, confianza promedio/mín/máx)
   - Parámetros de procesamiento v2.3
4. **Caracteres Segmentados:** Cada dígito/letra con:
   - Código de colores (verde ≥70%, naranja 55-70%, rojo <55%)
   - Borde rojo para caracteres débiles
   - Porcentaje de confianza individual

**Características del Script Mejorado (v2.3):**
- ✅ Incluye TODOS los caracteres (incluso débiles)
- ✅ Visualización de caracteres débiles con indicadores visuales
- ✅ Panel de análisis integrado con estadísticas
- ✅ Compatible con pipeline `visualize_sv.py`

---

### **3. Evaluación Masiva (Batch Test)**

Procesar múltiples imágenes desde estructura de carpetas con estadísticas:

```bash
python scripts/batch_test.py
```

**Comportamiento:**
- Busca recursivamente todas las imágenes (*.jpg) en `data/raw/`
- Procesa hasta 100 imágenes
- Extrae automáticamente el nombre del estado desde la ruta
- Calcula estadísticas globales de confianza

**Salida de ejemplo v2.3:**
```
--- 🚀 Iniciando Evaluación Masiva (MEJORADO v2.3) ---
Imágenes detectadas en subcarpetas: 87

ESTADO           | PREDICCIÓN   | CONF   | ARCHIVO
---------------------------------------------------------------------------
1_EL _SALVADOR   | P18183       | 89.2%  | placa_001.jpg
1_EL _SALVADOR   | P2701        | 76.5%  | placa_002.jpg
ALABAMA          | AL123456     | 91.3%  | placa_003.jpg

📊 ESTADÍSTICAS FINALES:
  • Total procesadas: 87
  • Confianza promedio: 84.3%
  • Confianza baja (<55%): 3 (3%)

💡 INTERPRETAR RESULTADOS:
  • CONF Alta (>70%): Predicción muy confiable
  • CONF Media (55-70%): Predicción aceptable
  • CONF Baja (<55%): Caracteres débiles dentro de la placa

✅ Prueba completada. Revisa predicciones con confianza baja para mejorar segmentación.
```

**Mejoras v2.3:**
- ✅ Incluye todos los caracteres (incluso débiles)
- ✅ Muestra confianza individual de cada predicción
- ✅ Estadísticas de caracteres débiles
- ✅ Mejor diagnóstico de problemas de segmentación

---

### **4. Visualización del Pipeline Completo**

Generar visualizaciones detalladas del proceso de OCR:

```bash
python scripts/visualize_sv.py
```

**Genera:**
- Imágenes preprocesadas (Black-Hat + Otsu)
- Regiones segmentadas de caracteres
- Predicciones individuales con confianza
- Resultado final anotado

**Salida:** Imágenes guardadas en `output_visuals/`

---

## 📊 Estructura de Datos

### **Dataset de Entrenamiento (Sintético v2.0)**
```
data/processed/train/
├── 0/ → 1000 imágenes (96×96, con múltiples transformaciones)
├── 1/ → 1000 imágenes
├── ...
├── 9/ → 1000 imágenes
├── A/ → 1000 imágenes
├── B/ → 1000 imágenes
└── Z/ → 1000 imágenes

Total: 36 clases × 1000 muestras = 36,000 imágenes
Mejora: 4x más datos, con augmentación agresiva
```

### **Dataset de Validación (Real)**
```
data/raw/valid/
├── 1_EL _SALVADOR/ → Placas salvadoreñas originales
├── ALABAMA/ → Placas de referencia (formato diferente)
└── ... (otros estados para comparación)
```

---

## 🔍 Detalles Técnicos

### **Preprocesamiento de Imagen (v2.2)**
- **Desenfoque Gaussiano:** Kernel 3×3 para reducir ruido metálico
- **Kernel Black-Hat:** Adaptativo (3.8% del ancho de imagen)
- **Umbralización:** Método de Otsu (adaptativo, sin parámetros manuales)
- **Limpieza Morfológica:** 
  - **Apertura:** Erosión + Dilatación (remueve ruido pequeño)
  - **Cierre:** Dilatación + Erosión (cierra gaps en caracteres reales)
  - **Kernels:** 2×2 para apertura, 3×3 para cierre
- **Padding Dinámico:** 18% vertical, 40% horizontal
- **Tamaño normalizado:** 96×96 píxeles (entrada ResNet50)

### **Segmentación de Caracteres (v2.2)**
| Parámetro | Valor |
|-----------|-------|
| Área mínima | 0.15% del área total |
| Altura | 25-75% de altura de imagen |
| Aspect Ratio | 0.12 - 1.15 |
| Banda Vertical | 30-70% de altura |
| Poda Máxima | 8 caracteres (formato VMT) |
| Erosión Pre-contornos | Kernel 2×2, 1 iteración |

### **Modelo ResNet50 (v2.0)**
- **Pesos iniciales:** ImageNet pre-entrenado
- **Entrada:** 96×96×3 (RGB)
- **Preprocesamiento:** Centrado en cero (modo Caffe de ResNet50)
- **Capas adicionales:**
  - GlobalAveragePooling2D
  - Dense(512, ReLU)
  - **BatchNormalization** ← NUEVO
  - Dropout(0.5)
  - Dense(36, Softmax)

### **Hiperparámetros de Entrenamiento (v2.0)**
| Fase | Épocas | Learning Rate | Frozen Layers | Augmentación |
|------|--------|---------------|---------------|--------------|
| Warm-Up | 10 | 0.001 | Base (ResNet50) | ON (dinámico) |
| Fine-Tuning | N | 0.00005 | Primeras 30 | ON (dinámico) |

---

## 🆕 Instrucciones para Reentrenamiento (v2.0)

Para aprovechar las mejoras de precisión, sigue este flujo:

### **Paso 1: Generar Dataset Mejorado**
```bash
python src/data/generator.py
```
**Resultado:**
- ✅ 36,000 imágenes sintéticas (1000/clase)
- ✅ Con augmentación agresiva y fondos variados
- ⏱️ Tiempo: ~3-5 minutos

### **Paso 2: Entrenar el Modelo Mejorado**
```bash
python src/models/train.py
```
**Resultado:**
- ✅ Modelo con BatchNormalization integrado
- ✅ Augmentación dinámica durante entrenamiento
- ✅ Learning rate adaptativo
- ⏱️ Tiempo: ~15-30 minutos (depende de GPU)

### **Paso 3: Evaluar Resultados**
```bash
python src/models/evaluate.py
```
**Resultados:**
- 📊 Accuracy general en validación
- 🔥 Matriz de confusión (PNG guardado)
- ⚠️ Pares de caracteres problemáticos
- 📈 Reporte por clase

### **Paso 4: Probar en Producción**
```bash
python scripts/run_inference.py --input "ruta/imagen.jpg"
```
**Salida mejorada (v2.3):**
```json
{
  "text": "SV123456",
  "confidence": 0.95,
  "min_confidence": 0.87,
  "valid_format": true,
  "weak_chars": []
}
```

**Ejemplo con carácter débil (v2.3):**
```json
{
  "text": "P18183",
  "confidence": 0.78,
  "min_confidence": 0.32,
  "valid_format": true,
  "weak_chars": [4]
}
```
Interpretación: Carácter en posición 4 tiene confianza <55% (32%) pero se incluye en resultado final.

---

## 🎓 Mejoras Esperadas

| Métrica | v2.0 | v2.1 | v2.2 | v2.3 |
|---------|------|------|------|------|
| Dataset | 36,000 img | ✓ | ✓ | ✓ |
| Augmentación | Agresiva | ✓ | ✓ | ✓ |
| BatchNorm | Sí | ✓ | ✓ | ✓ |
| Segmentación | Flexible | ✓ | Restrictiva v2.2 | ✓ |
| Validación | Patrón SV | ✓ | Flexible | Flexible |
| Caracteres Débiles | No | No | Filtrados | Incluidos |
| Accuracy (Carácter) | >90% | ✓ | >91% | >92% |
| Accuracy (Placa) | >85% | ✓ | >87% | >89% |

**Mejoras clave v2.2-v2.3:**
- ✅ Reducción de caracteres fantasma (segmentación restrictiva)
- ✅ Inclusión de caracteres débiles (mejor completitud)
- ✅ Visualización mejorada con panel de análisis
- ✅ Diagnóstico claro de caracteres problemáticos

---

## ⚠️ Consideraciones Importantes

1. **Calidad de imágenes:** El sistema funciona mejor con imágenes de 600px+ de ancho
2. **Reflejos metálicos:** El filtro Black-Hat está optimizado para capturar placas reflectantes
3. **Memoria:** Procesamiento de batch requiere ~1GB RAM por 100 imágenes
4. **GPU opcional:** Sin GPU, la inferencia toma ~2-3 segundos por imagen

---

## 📈 Métricas de Desempeño Esperadas

| Métrica | Valor |
|---------|-------|
| **Precisión (Caracter)** | >90% |
| **Precisión (Placa Completa)** | >85% |
| **Tiempo de Inferencia** | 2-3 segundos (CPU) / 200ms (GPU) |
| **Memoria Modelo** | ~95 MB |

---

## 🐛 Troubleshooting

### **Problema:** `ModuleNotFoundError: No module named 'src'`
**Solución:** Asegúrate de ejecutar scripts desde la raíz del proyecto y que PYTHONPATH incluya `src/`

### **Problema:** `Model not found: models/trained_model.h5`
**Solución:** Entrena primero con `python src/models/train.py` o descarga el modelo pre-entrenado

### **Problema:** Imágenes no detectadas en batch_test.py
**Solución:** Verifica que las imágenes estén en `data/raw/valid/` con estructura correcta de carpetas

### **Problema:** Bajo rendimiento en placas reales
**Solución:** Ajusta parámetros de preprocessing (kernel size, padding) en `src/inference/predict.py`

---

## 📞 Soporte y Contribuciones

Para reportar problemas o sugerencias, revisa los logs en el terminal y valida:
- ✓ Ruta del modelo existe
- ✓ Imágenes en formato JPG/PNG válidas
- ✓ Entorno virtual activado
- ✓ TensorFlow 2.10+ instalado

---

## 📝 Licencia

Este proyecto fue desarrollado originalmente con fines educativos y de investigación para el análisis de la brecha de dominio en matrículas salvadoreñas. 

El código fuente y los modelos liberados en este repositorio están bajo la licencia **GNU General Public License v3.0 (GPL-3.0)**. 

Esto significa que eres libre de usar, modificar y distribuir este software, bajo la estricta condición de que cualquier trabajo derivado o versión modificada también debe ser de código abierto y distribuirse bajo esta misma licencia.

Para más detalles, consulta el archivo [LICENSE](LICENSE) incluido en este repositorio.
---

**Última actualización:** Mayo 2026  
**Versión:** 2.3 - Inclusión de Caracteres Débiles y Visualización Mejorada