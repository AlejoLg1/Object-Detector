import cv2
import numpy as np
import time

def main():
    """
    Aplicación para la detección de objetos en tiempo real utilizando YOLOv3.

    Esta aplicación captura video desde una cámara en vivo y utiliza YOLOv3 para detectar objetos en cada cuadro.
    Los objetos detectados se dibujan con cuadros delimitadores y etiquetas de clase en el video en tiempo real.
    """
    # Cargar el modelo YOLOv3 preentrenado
    model = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")

    # Cargar las clases del conjunto de datos COCO
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los nombres de las capas de salida del modelo
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    # Generar colores aleatorios para las etiquetas de clase
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Configurar la captura de video desde la cámara
    camera = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    # Ajuste de los hiperparámetros
    confidence_threshold = 0.5  # Umbral de confianza mínima
    nms_threshold = 0.4  # Umbral de solapamiento de cuadros

    while True:
        # Capturar un cuadro de la cámara
        _, frame = camera.read()
        if frame is not None:
            frame_id += 1
            height, width, _ = frame.shape

            # Preprocesar el cuadro de la cámara
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            model.setInput(blob)
            outs = model.forward(output_layers)

            # Inicializar listas para almacenar información de detección
            class_ids = []
            confidences = []
            boxes = []

            # Procesar las salidas de las capas de detección
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        # Calcular coordenadas del cuadro delimitador
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Aplicar supresión no máxima
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

            # Dibujar cuadros delimitadores y etiquetas de clase
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255, 255, 255), 3)

            # Calcular y mostrar FPS
            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

            # Mostrar el cuadro de la cámara con detecciones
            cv2.imshow("Objetos detectados", frame)

            # Salir del bucle al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("No se pudo leer un cuadro de la cámara.")

    # Liberar recursos y cerrar ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
