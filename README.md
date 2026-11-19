# Acceso facial para cuatro personas en Raspberry Pi

Sistema de investigación configurado para tus dos datasets. Incluye ANN, CNN,
SVM, KNN, Random Forest y árbol de decisión. Cada uno identifica a las mismas
cuatro personas y se conecta al mismo control de acceso. Cada algoritmo aprende
también a clasificar rostro humano frente a objeto. Se usa **un algoritmo
por ejecución**, para poder compararlos por separado.

El software está en modo simulación por defecto. Los errores reales se obtienen
ejecutando el entrenamiento y la evaluación con los datasets.

## Preparación con tus carpetas actuales

`config.json` ya contiene `alexis_lopez`, `carol_canizares`, `jose_manzano` y
`manuel_ramirez`. Los originales permanecen intactos:

```text
dataset/
  Human faces and object dataset/Human faces and object dataset/Images/
    male_faces/      # humano
    female_faces/    # humano
    Objects/         # objeto
  webcam/
    alexis_lopez/sesion1/ ... sesion5/
    carol_canizares/sesion1/ ... sesion5/
    jose_manzano/sesion1/ ... sesion5/
    manuel_ramirez/sesion1/ ... sesion5/
```

Con `fuentes.reparto_identidad: "mezclado"`, se barajan las fotos de todas las
sesiones de cada persona con semilla fija antes de repartirlas:

| Partición de identidad | Fotos de cada persona |
|---|---|
| Entrenamiento | 60 % |
| Validación | 20 % |
| Prueba | 20 % |

Las fotos originales no se mueven. `fuentes.sesiones` enumera las sesiones que se
leen; sus grupos no deciden el destino en modo mezclado. Registra allí las sesiones
nuevas antes de preparar de nuevo. Para separar sesiones completas usa
`fuentes.reparto_identidad: "sesiones"` (el modo por defecto si se omite la opción).
El reparto mezclado puede colocar fotogramas muy parecidos en train, val y test;
sus métricas no demuestran generalización a nuevas sesiones. Mezclar no elimina
por sí solo asociaciones con lentes, fondos o iluminación.
Human faces se divide 60/20/20 por clase y semilla fija. Male y female se unen
como **humano**; no se entrena género. Se usan las carpetas, porque el CSV contiene
rutas absolutas del ordenador del autor del dataset.

Con el entorno virtual activo:

```bash
python app.py preparar
python app.py entrenar --algoritmo todos
# Un algoritmo en ambas tareas:
python app.py entrenar --algoritmo svm
# Solo una tarea:
python app.py entrenar --algoritmo todos --tarea humanos
python app.py entrenar --algoritmo todos --tarea identidad
```

`preparar` crea `artefactos/particiones.json` con rutas, etiquetas, sesiones y
hashes, sin copiar fotos. Omite duplicados exactos de archivos/píxeles y pone en
cuarentena las imágenes con etiquetas humano/objeto contradictorias. Registra
ambos casos en el manifiesto. Un duplicado entre personas o tareas detiene la
preparación. Entre sesiones también la detiene en modo sesiones; en modo mezclado
se conserva una sola copia por persona antes de repartir. No detecta todos los
duplicados aproximados.

`entrenar` prepara automáticamente si no hay manifiesto. Verifica los hashes
antes de reutilizar `datos_cache.joblib`. Si cambias las fotos, ejecuta `preparar`;
si cambias los modelos de visión o el tamaño mínimo de rostro, recalcula la caché.
Las imágenes de identidad sin un único rostro completo y de tamaño suficiente se
registran en `calidad_datos.json`, con ruta, motivo y conteos por persona/partición.
El entrenamiento se detiene si quedan menos de 3 muestras por persona en train o
menos de 1 en val/test. Las exclusiones cuentan como rechazos en el error de acceso
incluyendo calidad y en el error de la cascada completa.

En humano/objeto, CNN usa RGB de 64×64 y los otros cinco algoritmos usan HOG sobre
64×64. En identidad, CNN usa recortes alineados y los otros cinco usan SFace. Los
escaladores y aumentos se ajustan/aplican solo en entrenamiento. La cámara usa
el filtro humano/objeto del mismo algoritmo antes del reconocimiento.

Se imprimen errores de entrenamiento, validación y prueba de los seis algoritmos
en **ambas tareas**, y una tabla final. Resultados:

- `humanos_<algoritmo>.joblib` y `.json`: modelo binario, errores, humanos
  rechazados, objetos aceptados y matriz de confusión.
- `<algoritmo>.joblib` y `.json`: modelo de identidad y métricas de reconocimiento.
- Al entrenar ambas tareas: error de humano + identidad sobre webcam, incluyendo
  las exclusiones de calidad. El error de clasificación se mide en rostros válidos.
- `comparacion.json` y `comparacion.csv`: resultados de la ejecución actual.
  El CSV expresa errores entre 0 y 1; la consola los muestra en porcentaje.
- `resumen_errores.txt`: las tablas de errores en porcentaje, listas para leer.
- `entrenamiento.log`: salida de la ejecución realizada durante la preparación.

Human faces no incluye identificadores ni sesiones: su partición por imagen única
**no garantiza independencia por persona**. No se utiliza como desconocidos de
identidad. Para medir FAR hacen falta otras personas con IDs y sesiones separados
entre validación y prueba. Con las fuentes actuales FAR es `null` (no medida) y
GPIO sigue bloqueado.

Humano/objeto aprende rostros frente a objetos; no detecta cuerpos completos ni
presencia física. Una fotografía de un rostro puede clasificarse como humano.
No sustituye al PAD. El encuadre y los fondos de webcam pueden diferir del dataset
público; revisa el error de la cascada. Selecciona modelos con validación y reserva
test para informar; si ajustas después de mirar test, recoge un test nuevo.

## Instalación

Python 3.11 o superior. En Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cnn.txt
python app.py descargar-modelos
python app.py preparar
```

En Raspberry Pi OS de 64 bits, con Python y entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
# También si usarás CNN o el PAD entrenado aquí:
python -m pip install -r requirements-cnn.txt
python app.py descargar-modelos
python app.py preparar
```

PyTorch necesita una rueda compatible con tu versión de Python y arquitectura.
Si no está disponible para la Pi, entrena en un ordenador compatible y usa un
modelo PAD ONNX en la Pi; CNN necesita PyTorch también en inferencia. Los otros
cinco algoritmos funcionan sin PyTorch. Usa las mismas versiones de librerías al
trasladar artefactos joblib entre equipos. En una Pi sin escritorio sustituye
`opencv-python` por `opencv-python-headless` y usa `--sin-ventana`; no instales
ambos paquetes OpenCV en el mismo entorno.

Los modelos públicos YuNet y SFace se descargan de OpenCV Zoo. Los hashes usados
quedan en `artefactos/modelos.json`; cada modelo entrenado vincula esos hashes
para evitar usar otros descriptores por accidente. El comando de descarga no
sobrescribe modelos existentes y requiere Internet. Las dependencias instaladas
en el equipo de desarrollo figuran en `requirements-lock-desarrollo.txt`.

## Dataset de identidad

Esta sección describe el **modo anterior**, disponible para configuraciones sin
`fuentes`. Con tu configuración actual se leen las carpetas originales anteriores;
las carpetas vacías `dataset/train`, `val` y `test` no se utilizan.

Edita primero los cuatro identificadores de `autorizados` en `config.json`.
Los nombres de carpetas deben coincidir exactamente. Después ejecuta `preparar`.

```text
dataset/
  train/
    persona_1/sesion_01/*.jpg
    persona_2/sesion_01/*.jpg
    persona_3/sesion_01/*.jpg
    persona_4/sesion_01/*.jpg
  val/
    persona_1/sesion_02/*.jpg
    persona_2/sesion_02/*.jpg
    persona_3/sesion_02/*.jpg
    persona_4/sesion_02/*.jpg
    desconocidos/visitante_A/sesion_02/*.jpg
  test/
    persona_1/sesion_03/*.jpg
    persona_2/sesion_03/*.jpg
    persona_3/sesion_03/*.jpg
    persona_4/sesion_03/*.jpg
    desconocidos/visitante_B/sesion_03/*.jpg
```

Cada imagen debe contener exactamente un rostro, completo y de al menos 100 px
por lado (configurable). Se aceptan JPG, PNG y BMP. Las imágenes inválidas
interrumpen el entrenamiento y se informa su ruta: no se eliminan silenciosamente.
El mínimo técnico es 3 imágenes por persona en train y 1 en val/test; es solo
para comprobar el funcionamiento. Para una evaluación útil recoge muchas más,
en distintos días, iluminación, gafas, distancias y poses del reto.

Separa **sesiones de captura enteras** entre train, val y test. No repartas
fotogramas vecinos del mismo video entre particiones ni pongas aumentos de una
misma foto en particiones diferentes. El cargador rechaza sesiones con el mismo
nombre para una misma persona entre particiones y duplicados exactos de archivos
o píxeles. No detecta todos los duplicados aproximados: debes preparar bien los
datos. Usa desconocidos diferentes para calibración y test e incluye personas
parecidas a los autorizados. Nunca uses las cuatro identidades como desconocidos.

## Entrenamiento y comparación

```bash
python app.py entrenar --algoritmo todos
# O uno individual:
python app.py entrenar --algoritmo cnn
python app.py entrenar --algoritmo svm
```

Las seis opciones son `ann`, `cnn`, `svm`, `knn`, `random_forest` y
`arbol_decision`. ANN es un perceptrón multicapa. CNN es una red convolucional
PyTorch que aprende sobre recortes RGB alineados de 64×64; puede necesitar muchos
más datos y potencia. Los otros cinco modelos usan descriptores SFace normalizados
y la misma partición. Por tanto la comparación es entre sistemas completos con
esas representaciones, no una comparación aislada de clasificadores sobre las
mismas entradas. Todos comparten el mismo filtro de identidad abierta.

Se imprime para cada algoritmo:

- Error de clasificación en **train, val y test**, sobre las cuatro personas.
- Brecha = error de test − error de train, en puntos porcentuales.
- Error del sistema de acceso, contando rechazos e identificaciones incorrectas.
- Tasa de aceptación de desconocidos (FAR empírica) y rechazo de conocidos.

Se guardan el modelo en `artefactos/<algoritmo>.joblib`, los resultados y matrices
de confusión en `<algoritmo>.json` y la comparación en `comparacion.json`. Una FAR
`null` significa que no había desconocidos para medirla. Una brecha negativa es
posible y no se convierte a valor absoluto. ANN puede mostrar una advertencia de
convergencia; revisa los datos y el número de iteraciones antes de confiar en ella.

La calibración usa solamente val: busca umbrales de similitud y margen que no
acepten desconocidos de esa partición, respetando los mínimos configurados.
Sin desconocidos se usan esos mínimos y GPIO permanece bloqueado. Test se usa
para informar el resultado y comprobar requisitos mínimos de despliegue, nunca
para elegir umbrales. Para seleccionar un algoritmo usa val; si eliges o ajustas
tras mirar test, recoge un nuevo test independiente para la evaluación final.

## Detección de fotos, pantallas y videos (PAD)

El acceso requiere simultáneamente identificación consistente, similitud con la
plantilla del autorizado, margen frente a las otras tres personas, PAD positivo
en cada fotograma y un reto aleatorio de movimientos sostenidos. Un fallo, cambio
de persona, varios rostros o una pérdida de detección reinicia la sesión.

Las cuatro carpetas de identidad **no entrenan por sí solas** un detector de
suplantación. Puedes entrenar el PAD binario incluido con capturas de la cámara:

```text
dataset_pad/
  train/real/captura_train/*.jpg
  train/ataque/captura_train/*.jpg
  val/real/captura_val/*.jpg
  val/ataque/captura_val/*.jpg
  test/real/captura_test/*.jpg
  test/ataque/captura_test/*.jpg
```

`real` contiene personas vivas frente a la cámara. `ataque` contiene imágenes
**capturadas por la cámara** de impresiones, móviles, monitores y reproducción
de video; no basta con guardar archivos de fotos originales en esa carpeta.
Separa también dispositivos de ataque, personas y condiciones entre particiones.

```bash
python app.py entrenar-pad --datos dataset_pad --epocas 30
```

Este comando entrena otra CNN, calibra solo en val y genera `artefactos/pad.joblib`
y `pad.json` con error train/val/test, brecha, APCER (ataques aceptados) y BPCER
(personas reales rechazadas). Copia a `config.json` la ruta `pad.modelo` y el valor
exacto de `pad.umbral` que imprime el comando. El PAD local usa recorte de rostro
con contexto 1.5×, RGB 64×64 y valores 0–1; no requiere ONNX. Si todos los reales
son rechazados, ese PAD no es útil aunque APCER sea cero: mejora datos/modelo.

También se admite un modelo **ONNX binario ya entrenado**. Configura en `pad`
su ruta, tamaño, escala, media, orden RGB/BGR, si devuelve logits o probabilidades,
el índice de clase real y el umbral. El adaptador espera entrada NCHW de un rostro
con contexto 1.5× y salida de exactamente dos valores. Comprueba el contrato del
modelo elegido; no todos los modelos PAD usan ese recorte o normalización. La
transformación ONNX es `(píxel - media) * escala`, seguida de intercambio BGR/RGB
si corresponde. No se incluye un modelo PAD preentrenado de terceros ni se asume
que cualquier ONNX sea compatible. Carga solo artefactos joblib propios y fiables:
pueden ejecutar código al deserializarse.

**Límite importante:** PAD y movimiento con cámara RGB pueden ser burlados por
ataques sofisticados; no prueban profundidad y no garantizan diferenciar gemelos
o todos los parecidos. El reto usa la posición de la nariz respecto a los ojos,
no una estimación 3D de postura. Prueba las direcciones mostradas con tu cámara;
no cambies el reflejo horizontal del video sin comprobarlas. Para una puerta con
exigencia de seguridad real usa sensor de profundidad/IR y un segundo factor,
además de validar el sistema completo con ataques no vistos. El segundo factor
y un sensor IR/profundidad no están implementados en este prototipo RGB.

## Cámara y puerta

```bash
# Simulación: nunca usa GPIO; sin PAD deniega todo.
python app.py ejecutar --algoritmo svm --camara 0
# Pi sin escritorio:
python app.py ejecutar --algoritmo svm --camara 0 --sin-ventana
```

Usa una cámara USB/V4L2 compatible con `cv2.VideoCapture`. La cámara CSI de la Pi
necesita un puente V4L2/GStreamer compatible; no se incluye un backend Picamera2.
La tecla Q, Ctrl+C o SIGTERM detienen el bucle y desactivan el relé. El registro
`artefactos/accesos.jsonl` contiene cambios de estado, identidad y modo; no guarda
fotogramas. Los timestamps están en UTC. Protege y limita la retención de datasets,
modelos biométricos y registros.

Para permitir GPIO, primero valida con el montaje real el reconocimiento de las
cuatro personas, desconocidos parecidos, impresiones, pantallas, videos y pérdidas
de cámara. El artefacto debe haber rechazado todos los desconocidos de val/test y
aceptado al menos una muestra de cada autorizado en ambas particiones. Esto es un
filtro mínimo técnico, **no certifica la seguridad** ni impone una tasa máxima de
rechazo de reales. Establece criterios de aceptación y tamaño muestral adecuados
para tu instalación; cero errores en pocas imágenes no demuestra riesgo cero.

Tras esa validación, configura `despliegue_validado: true` como declaración
explícita de que hiciste las pruebas y ejecuta:

```bash
python app.py ejecutar --algoritmo svm --camara 0 --hardware --sin-ventana
```

También se exige que haya un PAD cargado. La declaración de despliegue es manual;
el programa no puede comprobar que ensayaste todos los ataques. Configura
`rele.pin_bcm` (numeración BCM, 17 por defecto), `activo_alto`, duración y
enfriamiento. Prueba primero con una carga de prueba. El GPIO solo gobierna un
driver/relé compatible con lógica 3.3 V; la cerradura necesita su propia fuente y
protección inductiva. No la conectes directamente al GPIO. Selecciona un estado
eléctrico de reposo apropiado también durante arranque/apagado y una salida de
emergencia independiente. El software apaga el relé en `finally`; no puede
garantizar su estado ante cortes de corriente, fallos del proceso o del hardware.

## Verificación del código

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan datos sintéticos para comprobar entrenamiento y persistencia de
los seis algoritmos y del PAD, rechazo de desconocidos, calibración, separación de
sesiones, detección de duplicados, reto, reinicios y cierre del relé ante
excepciones. También se comprueba el rechazo de una imagen vacía con los modelos
públicos cuando están descargados. No sustituyen la evaluación biométrica ni
prueban la cámara o los pines físicos.

## Referencias

- [OpenCV: detección YuNet, alineación y reconocimiento SFace](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html)
- [Modelos y licencias de OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- [scikit-learn: pipelines de preprocesamiento y clasificación](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [GPIO Zero: dispositivos de salida](https://gpiozero.readthedocs.io/en/stable/api_output.html)
