# Memoria técnica — Reconocimiento de cartas (visión clásica)

## Hardware
- Cámara: 720p–1080p, 30 fps; óptica fija, enfoque automático desactivable.
- Iluminación: homogénea, difusa; evitar reflejos fuertes sobre el naipe.
- Escenario: tapete verde uniforme cubriendo todo el fondo.
- Justificación: resolución suficiente para detalles en la esquina; fondo controlado simplifica segmentación y reduce falsos positivos.

## Software
- Lenguaje: Python 3.x
- Librerías: OpenCV (`cv2`), NumPy
- Estructura del proyecto: `src/` con módulos de detector, utilidades, reconocedor y CLI; plantillas en `assets/templates`, muestras en `cartas/`.
- Justificación: OpenCV/NumPy permiten procesamiento clásico (colores, contornos, morfología, correlación) cumpliendo la restricción de no usar aprendizaje.

## Hoja de ruta
- Detector por fondo verde y “blancura de carta” para recorte robusto.
- Warp de perspectiva y orientación automática del naipe.
- Extracción de esquina superior-izquierda y realce local (CLAHE/gamma).
- Reconocimiento de rango por plantillas y de palo por color HSV + plantilla.
- Fallback por correlación del warp contra `cartas/` si la confianza es baja.
- HUD estable en esquina superior izquierda y vista comparativa con el recorte.

## Solución
- Pipeline general:
  - Segmentar tapete en HSV y extraer regiones no verdes.
  - Intersectar con máscara de “blanco” en LAB para aislar el rectángulo de la carta.
  - Contornos → cuadrilátero (aprox o `minAreaRect`) → `four_point_transform`.
  - Probar 4 orientaciones del warp; elegir la mejor según similitud de rango.
  - Extraer la esquina, realzarla, separar rango/palo por componentes.
  - Rango: correlación + forma con `assets/templates/ranks`.
  - Palo: grupo por color (rojo/negro, HSV) + plantilla en `assets/templates/suits`.
  - Si la confianza es insuficiente, comparar el warp con las imágenes de `cartas/`.
  - Dibujar etiqueta estable y resaltar carta; opción de vista comparativa.

## Diagrama de decisión (texto)
- Para cada carta detectada:
  - ¿Hay warp con rectángulo válido y suficiente “blancura”?
    - No → descartar región.
    - Sí → orientar el warp (mejor de 4 rotaciones).
  - ¿Scores de rango/palo ≥ umbrales y margen entre top-1 y top-2 ≥ `margin`?
    - Sí → etiquetar con `rank` y `suit`.
    - No → fallback por corner completo; si sigue bajo → fallback `CardDB.match(warp)`.
  - ¿Color rojo > umbral? → limitar candidatos a corazones/diamantes; si no, picas/trébol.
  - Mostrar etiqueta estable vía voto temporal y resaltar polígono de carta.

## Secuencialización y parámetros
- Segmentación de tapete y carta: `src/detector.py:27-43`
  - HSV verde dinámico con `_estimate_green_range`: `src/detector.py:5-25` (margen ±20 en H; S,V mínimos).
  - Máscara LAB “blanco” (L ∈ [170,255], |A−128|, |B−128| ≤ 35): `src/detector.py:36-43`.
  - Morfología `OPEN/CLOSE` con `kernel=5×5` para limpiar ruido: `src/detector.py:43-45`.
- Contornos y cuadrilátero: `src/detector.py:49-75`
  - Filtro de área mínima `0.01·W·H` para evitar pequeños artefactos: `src/detector.py:51-52`.
  - `approxPolyDP(ε=0.02·perímetro)` o `minAreaRect` si no hay 4 vértices: `src/detector.py:54-67`.
  - Relación de aspecto 1.25–1.75 para cartas; `solidity ≥ 0.5`: `src/detector.py:63-74`.
  - Validación de “blancura” dentro del warp con ratio ≥ 0.35: `src/detector.py:68-75`.
- Warp de perspectiva: `src/utils.py:14-19`
  - Orden de puntos y `cv2.getPerspectiveTransform`; tamaño estándar `(w=300,h=420)`.
- Orientación del warp: `src/recognizer.py:56-73`
  - Rotar 0/90/180/270 y elegir por máximo score de rango sobre corner.
- Realce local: `src/utils.py:28-37` y uso: `src/recognizer.py:109-114`
  - CLAHE (`clipLimit=2.0`) sobre V; gamma dinámico según brillo medio del corner.
- Esquina y separación rango/palo: `src/recognizer.py:75-106`
  - Extracción del rectángulo 2–30% ancho × 2–25% alto del warp: `src/recognizer.py:76-80`.
  - Otsu y componentes; se usan cajas superior e inferior si hay ≥2 componentes.
- Rango (templates): `src/recognizer.py:125-138`
  - `cv2.matchTemplate(TMQ=CCOEFF_NORMED)` + similitud de forma (`matchShapes`); peso forma 0.3.
  - Umbral de aceptación `rank_min=0.45` y margen `margin=0.05` entre top-1 y top-2.
- Palo por color + templates: `src/recognizer.py:155-173`
  - Umbral de rojo HSV por área (`red_ratio>0.03`) decide grupo; dentro del grupo se usa plantilla; peso forma 0.7.
  - `suit_min=0.5`, mismo margen.
- Fallback por correlación con `cartas/`: `src/card_db.py:32-44`, `src/recognizer.py:194-201`
  - `cv2.matchTemplate` sobre warp completo; si score ≥ 0.55 se adopta etiqueta de archivo.
- Visualización y estabilidad: `src/main.py:25-41`, `src/main.py:116-127`, `src/main.py:135-140`
  - HUD arriba-izquierda; resaltado de carta; ventana de voto temporal (`hold_frames`) y vista comparativa.

## Ejecución
- Cámara 1: `python -m src.main --camera 1 --compare_view --brightness 0.9 --hold_frames 9 --rank_thresh 0.45 --suit_thresh 0.5`
- Afinar verde: `--green_low 35,40,40 --green_high 85,255,255` (o valores según tu tapete).
- Imagen: `python -m src.main --input cartas\3-de-diamantes-roja.png --save salida.png --single --compare_view`

## Justificación de parámetros
- HSV ±20 en H y mínimos en S,V: tolera variaciones de tono del tapete por iluminación.
- LAB “blanco” con L alto y A,B cerca de 128: robusto frente a saturación y sombras.
- `approxPolyDP` con 0.02·perímetro: típico para aproximar rectángulos con bordes suaves.
- Relaciones de aspecto y `solidity`: descartan objetos no rectangulares y oclusiones fuertes.
- CLAHE/gamma: estabiliza contraste local del corner sin alterar geometría.
- Pesos de forma (0.3 rangos, 0.7 palos): palos dependen más de silueta; rangos más del patrón textual.
- Umbrales moderados (`rank_min=0.45`, `suit_min=0.5`) y margen `0.05`: balance entre precisión y respuesta en cámara.

