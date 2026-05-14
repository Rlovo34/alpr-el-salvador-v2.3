import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

class PlatePredictor:
    """Motor de predicción OCR para placas vehiculares salvadoreñas.
    
    Implementa un pipeline híbrido que combina visión clásica (OpenCV) 
    y aprendizaje profundo (ResNet50) para detectar y reconocer 36 clases 
    de caracteres (0-9, A-Z) en placas reflectantes del VMT de El Salvador.
    
    Arquitectura del pipeline:
        1. Preprocesamiento: Estandarización → Black-Hat → Otsu → Dilatación
        2. Segmentación: Extracción de contornos → Padding dinámico → 96x96
        3. Clasificación: ResNet50 fine-tuned → Softmax → Validación de patrón
    
    Atributos:
        model (tf.keras.Model): Modelo ResNet50 entrenado y persistido en H5.
        classes (list): Lista de 36 caracteres válidos en orden alfabético-numérico.
    """
    
    def __init__(self, model_path):
        """Inicializa el predictor cargando el modelo entrenado.
        
        Args:
            model_path (str): Ruta absoluta o relativa al archivo del modelo (.h5).
            
        Raises:
            FileNotFoundError: Si el archivo del modelo no existe.
            ValueError: Si el modelo no es compatible con ResNet50.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    @staticmethod
    def validate_plate_format(predicted_text):
        """Valida que la predicción cumpla con el formato legal de placas SV.
        
        El VMT de El Salvador acepta dos variantes de formato:
        1. Estricta: ABC12345 (2 letras iniciales + 5 dígitos)
        2. Flexible: Mixta con ≥2 letras y ≥3 dígitos iniciando con letra
        
        MEJORADO v2.2: Flexible con longitud 6-8 caracteres
        - Antes: Exactamente 8 caracteres (muy restrictivo post-filtro confianza)
        - Ahora: 6-8 caracteres (tolera descartes por baja confianza)
        
        Esta validación rechaza imposibilidades lógicas como:
        - Placas de 8 números (sin letras)
        - Placas que no empiezan con letra
        - Longitud fuera del rango 6-8
        
        Args:
            predicted_text (str): Texto predicho por el modelo ResNet50.
            
        Returns:
            bool: True si cumple formato legal SV, False en caso contrario.
            
        Example:
            >>> PlatePredictor.validate_plate_format("SV1234")   # 6 chars
            True
            >>> PlatePredictor.validate_plate_format("SV123456") # 8 chars
            True
            >>> PlatePredictor.validate_plate_format("12345678") # Sin letras
            False
        """
        # MEJORADO: Permite 6-8 caracteres (vs exactamente 8)
        if len(predicted_text) < 6 or len(predicted_text) > 8:
            return False
        
        # Validar que tenga estructura de placa: letras al inicio, números al final
        # Variante 1: ABC12345 (2 letras + resto números)
        if len(predicted_text) >= 3 and predicted_text[:2].isalpha() and predicted_text[2:].isdigit():
            return True
        
        # Variante 2: A1B2C3... (mixto pero comenzando con letra)
        if predicted_text[0].isalpha():
            # Al menos 2 letras y 3 números
            letters = sum(1 for c in predicted_text if c.isalpha())
            numbers = sum(1 for c in predicted_text if c.isdigit())
            if letters >= 2 and numbers >= 3:
                return True
        
        return False

    def standardize_image(self, img):
        """Estandariza imagen a resolución canónica (600px ancho).
        
        Justificación técnica:
        - Resolución 600px: balance entre velocidad y precisión de segmentación
        - Mantiene relación de aspecto original (evita distorsión)
        - Usa INTER_AREA para reducir aliasing en downsampling
        
        Args:
            img (np.ndarray): Imagen BGR cargada con cv2.imread()
            
        Returns:
            np.ndarray: Imagen redimensionada preservando aspect ratio.
        """
        height, width = img.shape[:2]
        new_width = 600
        new_height = int((new_width / float(width)) * height)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def preprocess_image(self, image_path):
        """Fase 1: Preprocesamiento avanzado para extracción de caracteres.
        
        Técnicas aplicadas (en orden):
        1. Estandarización: 600px ancho (resolución canónica)
        2. Conversión escala de grises: Simplifica cálculos morfológicos
        3. Desenfoque Gaussiano: Reduce ruido de reflexión de metal
        4. Filtro Black-Hat: Extrae caracteres oscuros sobre fondo claro
           - Kernel dinámico = 3.8% del ancho (adaptable a resoluciones)
           - Ideal para placas reflectantes del VMT
        5. Umbralización Otsu: Binarización automática sin parámetros manuales
        6. Limpieza morfológica MEJORADA: Erosión + Dilatación para remover ruido
           - Erosión: Abre pequeños gaps en artefactos
           - Dilatación: Cierra gaps en caracteres reales
        
        Justificación matemática:
        - Black-Hat (closing - img) = extrae características pequeñas oscuras
        - Otsu (THRESH_BINARY): Encuentra umbral óptimo automáticamente
        - Kernel 3.8%: Calibrado empíricamente para caracteres alfanuméricos
        - Limpieza morfológica: Erosión + Dilatación = "Apertura" anti-ruido
        
        Args:
            image_path (str): Ruta a archivo de imagen (JPG/PNG).
            
        Returns:
            tuple: (thresh, img_std)
                - thresh (np.ndarray): Imagen binarizada lista para contornos
                - img_std (np.ndarray): Imagen estandarizada original (para visualización)
                
        Raises:
            ValueError: Si el archivo de imagen no existe o es inválido.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error cargando imagen: {image_path}")
        
        # 1. Estandarización
        img = self.standardize_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape[:2]
        
        # Opcional pero recomendado: Ligero desenfoque para reducir ruido del metal reflectivo
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. FILTRO BLACK-HAT: Extrae texto oscuro sobre fondo claro
        # Kernel adaptativo basado en dimensiones de imagen
        kernel_size = int(img_w * 0.038)  # ~3.8% del ancho
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
        
        # 3. UMBRALIZACIÓN DE OTSU
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 4. LIMPIEZA MORFOLÓGICA MEJORADA
        # Apertura: Erosión + Dilatación (abre/remueve ruido pequeño)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        
        # Cierre: Dilatación + Erosión (cierra gaps en caracteres reales)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_clean = cv2.morphologyEx(thresh_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                                           
        return thresh_clean, img

    def _format_char_image(self, char_roi, h):
        """Formatea regiones de interés (ROI) de caracteres para ResNet50.
        
        Proceso de normalización en 4 pasos:
        1. Padding dinámico (18% vertical, 40% horizontal)
           - 18% V: Separa caracteres de bordes, similar a proporciones reales
           - 40% H: Acomoda variaciones de ancho en caracteres (I vs W)
        2. Cuadratización: Crea lienzo cuadrado sin distorsión
           - Evita compresión horizontal/vertical al redimensionar
           - Mantiene proporciones naturales del carácter
        3. Redimensionamiento a 96x96: Entrada canónica de ResNet50
        4. Conversión BGR: Compatible con preprocesamiento ResNet50
        
        Justificación:
        - Padding 0.18/0.40 calibrado empíricamente en placas SV reales
        - Cuadratización es crítica: evita que "I" se vea como "1"
        - INTER_AREA: Mejor calidad que INTER_LINEAR para downsampling
        
        Args:
            char_roi (np.ndarray): Región de carácter aislado (escala de grises).
            h (int): Altura en píxeles del carácter original.
            
        Returns:
            np.ndarray: Imagen normalizada RGB 96x96 lista para ResNet50.
        """
        # Padding dinámico  mejor resultado 0.18 y 0.40 para SV
        pad_y = int(h * 0.18)
        pad_x = int(h * 0.40)
        padded_char = cv2.copyMakeBorder(char_roi, pad_y, pad_y, pad_x, pad_x, 
                                         cv2.BORDER_CONSTANT, value=0)
        
        # Creación del lienzo cuadrado (Evita la deformación al redimensionar)
        ph, pw = padded_char.shape
        sq_size = max(ph, pw)
        fondo = np.zeros((sq_size, sq_size), dtype=np.uint8)
        
        y_off = (sq_size - ph) // 2
        x_off = (sq_size - pw) // 2
        fondo[y_off:y_off+ph, x_off:x_off+pw] = padded_char
        
        # Redimensión final al formato de entrada de ResNet50
        char_resized = cv2.resize(fondo, (96, 96), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(char_resized, cv2.COLOR_GRAY2RGB)

    def extract_characters(self, thresh, img_original):
        """Fase 2: Segmentación de caracteres con filtros MEJORADOS.
        
        Estrategia: Filtros restrictivos pero adaptativos para captar solo caracteres válidos.
        MEJORAS v2.2:
        - Área mínima aumentada: 0.0015 (0.15% vs 0.05%) → menos ruido/reflexiones
        - Aspect ratio más estrecho: 0.12-1.15 (vs 0.10-1.30) → evita overfitting a ruido
        - Altura más controlada: 25-75% (vs 20-80%) → menos caracteres fantasma en bordes
        - Banda vertical: 30-70% (vs 25-75%) → centrado más estricto
        - Morfología mejorada: Erosión anti-ruido antes de contornos
        
        Filtros aplicados (en cascada):
        1. LIMPIEZA MORFOLÓGICA: Erosión para remover ruido/reflejos mínimos
           
        2. UMBRAL DE ÁREA: 0.15% del área total (aumentado de 0.05%)
           - Descarta reflexiones metálicas y artefactos
           - Dinámico según resolución de imagen
           
        3. RANGO DE ALTURA: 25-75% de altura de imagen (vs 20-80%)
           - Reduce capturas de ruido en bordes
           - Tolera variaciones moderadas de inclinación
           
        4. ASPECT RATIO: 0.12-1.15 (vs 0.10-1.30)
           - 0.12: Permite letras estrechas pero rechaza filamentos
           - 1.15: Tolera caracteres anchos pero rechaza manchas
           
        5. POSICIÓN VERTICAL: 30-70% de altura (vs 25-75%)
           - Restringe verticalmente para placas bien alineadas
           - Rechaza caracteres periféricos que suelen ser ruido
           
        6. PODA LÓGICA: Máximo 8 caracteres (formato VMT)
           - Si detecta >8, mantiene los 8 más grandes (mejor que aleatorio)

        Args:
            thresh (np.ndarray): Imagen binarizada de entrada.
            img_original (np.ndarray): Imagen original (BGR) para contexto.
            
        Returns:
            list[np.ndarray]: Lista de caracteres normalizados 96x96 RGB,
                              ordenados de izquierda a derecha.
        """
        # MEJORA: Erosión anti-ruido para eliminar pequeños artefactos
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh_clean = cv2.erode(thresh, kernel_erode, iterations=1)
        
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        characters = []
        img_h, img_w = img_original.shape[:2]
        
        # FILTRO 1: Umbral de área dinámico MEJORADO (0.15% vs 0.05%)
        area_threshold = img_h * img_w * 0.0015  # 0.15% del área total

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Descartar basura microscópica
            if area < area_threshold:
                continue
                
            aspect_ratio = w / float(h)
            center_y = y + (h / 2.0)

            # FILTRO 2: Geometría de Altura y Proporción MEJORADA
            # Rango de altura: 25% a 75% (vs 20-80%)
            # Aspect ratio: 0.12 a 1.15 (vs 0.10-1.30)
            if (img_h * 0.25) < h < (img_h * 0.75) and 0.12 < aspect_ratio < 1.15:
                
                # FILTRO 3: BANDA VERTICAL MEJORADA
                # Centro de gravedad entre 30% y 70% (vs 25-75%)
                if (img_h * 0.30) < center_y < (img_h * 0.70):
                    
                    char_roi = thresh_clean[y:y+h, x:x+w]
                    char_rgb = self._format_char_image(char_roi, h)
                    
                    characters.append((x, char_rgb, area))
                
        # PODA LÓGICA: Las placas SV tienen un máximo de 8 caracteres
        if len(characters) > 8:
            characters.sort(key=lambda item: item[2], reverse=True)
            characters = characters[:8]
            
        # Orden espacial de izquierda a derecha
        characters.sort(key=lambda item: item[0])
        
        return [char[1] for char in characters]

    def run(self, image_path, min_char_confidence=0.55):
        """Fase 3: Flujo de inferencia completo con inclusión de todos los caracteres.
        
        Orquesta las tres fases del pipeline:
        1. Preprocesamiento: Extrae máscara binaria
        2. Segmentación: Detecta regiones de caracteres
        3. Clasificación: Predice con ResNet50 todos los caracteres
        
        MEJORADO v2.3: 
        - TODOS los caracteres se incluyen en el resultado final
        - Sin filtro de confianza (el parámetro es solo informativo ahora)
        - Mantiene caracteres débiles (ej: 32%) para completar la placa
        
        Manejo de casos excepcionales:
        - Si no detecta caracteres: retorna "NO_DETECTADO" con confianza 0.0
        - Si detecta <8 caracteres: procesa los disponibles
        - Siempre valida que la predicción cumpla formato legal SV
        
        Cálculo de confianza:
        - confidence: Promedio de probabilidades máximas por carácter
        - min_confidence: Carácter menos confiable (cuello de botella)
        - valid_format: Boolean indicando si cumple formato VMT
        
        Args:
            image_path (str): Ruta a imagen de placa (JPG/PNG).
            min_char_confidence (float): Solo informativo, no filtra caracteres (0-1).
                                        Se usa para marcar caracteres débiles en visualización.
                                        Default: 0.55 (55%)
            
        Returns:
            dict: Estructura con 5 claves:
                {
                    "text": str,                    # Predicción con TODOS los caracteres
                    "confidence": float,            # Confianza promedio [0.0, 1.0]
                    "min_confidence": float,        # Confianza del carácter débil
                    "valid_format": bool,           # Cumple formato legal SV
                    "weak_chars": list              # Lista de índices de caracteres débiles (<55%)
                }
                
        Example:
            >>> predictor = PlatePredictor("models/trained_model.h5")
            >>> result = predictor.run("data/raw/valid/1_EL _SALVADOR/plate.jpg")
            >>> print(result["text"], result["confidence"])
            SV123456 0.92
        """
        thresh, img = self.preprocess_image(image_path)
        char_images = self.extract_characters(thresh, img)
        
        if not char_images:
            return {
                "text": "NO_DETECTADO", 
                "confidence": 0.0, 
                "min_confidence": 0.0,
                "valid_format": False,
                "weak_chars": []
            }

        # Preparación de lote (Batch) para ResNet50
        batch = np.array(char_images, dtype=np.float32)
        batch = preprocess_input(batch) 
        
        predictions = self.model.predict(batch, verbose=0)
        
        final_text = ""
        total_confidence = 0.0
        confidences = []  # Guardar confianzas individuales
        weak_chars = []   # Guardar índices de caracteres con baja confianza
        
        for idx, pred in enumerate(predictions):
            conf = np.max(pred)
            pred_class = self.classes[np.argmax(pred)]
            
            # v2.3: INCLUIR TODOS LOS CARACTERES (sin filtrar por confianza)
            final_text += pred_class
            total_confidence += conf
            confidences.append(conf)
            
            # Marcar caracteres débiles para visualización (solo informativo)
            if conf < min_char_confidence:
                weak_chars.append(idx)
        
        avg_confidence = total_confidence / len(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        
        # VALIDACIÓN DE PATRÓN: Verificar formato de placa SV
        is_valid_format = self.validate_plate_format(final_text)
        
        return {
            "text": final_text, 
            "confidence": avg_confidence,
            "min_confidence": min_confidence,
            "valid_format": is_valid_format,
            "weak_chars": weak_chars
        }