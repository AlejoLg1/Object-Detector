# Detector de objetos con YOLOv3

Este proyecto implementa un detector de objetos en tiempo real utilizando la arquitectura YOLOv3 y OpenCV. El detector es capaz de identificar y dibujar cuadros delimitadores alrededor de varios objetos en un flujo de video en vivo.

## Requisitos

- Python 3.x
- OpenCV
- NumPy

## Instalación

1. Clona o descarga este repositorio:

~~~
https://github.com/AlejoLg1/Object-Detector.git
~~~


2. Instala las dependencias necesarias:

~~~
pip install opencv-python numpy
~~~


## Uso

1. Ejecuta el script `main.py`:

~~~
python main.py
~~~


2. La aplicación comenzará a capturar video desde la cámara en vivo y detectará objetos en cada cuadro.

3. Para detener la aplicación, presiona la tecla 'q'.

## Personalización

- Puedes ajustar los hiperparámetros como el umbral de confianza (`confidence_threshold`) y el umbral de supresión no máxima (`nms_threshold`) en el archivo `main.py` según sea necesario para tu aplicación.
- Para usar tu propio modelo YOLOv3, asegúrate de actualizar los archivos de configuración y pesos en la función `cv2.dnn.readNetFromDarknet()`.

## Factores que afectan el rendimiento

El rendimiento de la aplicación puede variar significativamente según los componentes del sistema de la computadora del usuario. A continuación, se enumeran algunos factores importantes que pueden influir en la experiencia de uso:

- Procesador (CPU)
- Tarjeta gráfica (GPU)
- Memoria RAM
- Disco duro

## Imágenes

![Car][images\car.png]

![Botle][images\bottle.png]

![Remote][images\remote.png]


