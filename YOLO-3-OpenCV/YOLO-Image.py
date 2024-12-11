"""
Training YOLO v3 for Object Detection with Custom Data
"""

# Detecting Objects on Image with OpenCV deep learning library
#
# Algorithm:
# Reading RGB image --> Getting Blob --> Loading YOLO v3 Network -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels
#
# Result:
# Window with Detected Objects, Bounding Boxes and Labels

# Importar las librerias
import numpy as np
import cv2
import time

"""
Leer la imagen
"""
image_BGR = cv2.imread("./images/Sin-titulo.jpg")

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", image_BGR)
cv2.waitKey(0)
cv2.destroyWindow("Original Image")

# Checkpoint
# Show image shape
print()
print("Image shape: ", image_BGR.shape)  # tupla (511, 767, 3)

# Guardar las dimensiones
h, w = image_BGR.shape[:2]

# Mostrar alto y ancho de la imagen
print()
print(f"Image height = {h} and width = {w}")


"""
Getting blob de la imagen de entrada
"""
# blob es el preprocesado de la imagen despues de la sustraccion
# de la media, la normalizadion y el swap de los canales RB,
# resultado en un elemento con shape 4:
# Nº de imagens, Nº de canales, ancho y alto
blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)

print()
print("Image shape: ", image_BGR.shape)  # tupla (511, 767, 3)
print("Blob shape: ", blob.shape)  # tupla (511, 767, 3)

# Mostrar el blob (ha que transformarlo)
blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
print()
print("blob_to_show shape: ", blob_to_show.shape)  # tupla (511, 767, 3)

cv2.namedWindow("Blob Image", cv2.WINDOW_NORMAL)
cv2.imshow("Blob Image", cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyWindow("Blob Image")


"""
Cargar el modelo YOLO-3
"""
# Cargar los labels
with open("./yolo-coco-data/coco.names") as f:
    labels = [line.strip() for line in f]

print()
print("Lista con las etiquetas:")
print(labels)


# Cargar el modelo YOLO entrenado
network = cv2.dnn.readNetFromDarknet(
    "./yolo-coco-data/yolov3.cfg", "./yolo-coco-data/yolov3.weights"
)

# Obtener una lista con los nombres de las capas del YOLO v3
layers_names_all = network.getLayerNames()
print()
print(layers_names_all)

# Obtener solo unas capas que necesitamos del YOLOv3 (restar 1 por identar en 0)
layers_names_output = [
    layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()
]

print()
print(
    layers_names_output
)  # El resultado tiene que ser ['yolo_82', 'yolo_94', 'yolo_106']

# Definir la probabiliddad minima para evitar predicciones debiles
probability_minimum = 0.5

# Ajustar el threshold para filtrar los cajas
threshold = 0.3

# Generar los colores para representar cada objeto detectado
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

print()
print(type(colours))
print(colours.shape)
print(colours[0])


"""
Implementar el Forard pass
"""
network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# Mostrar el tiempo del forward pass
print("Objects Detection took {:05f} seconds".format(end - start))


"""
Obtener los bordes de las cajas
"""
bounding_boxes = []
confidences = []
class_numbers = []


# Recorrer todas las salidas despues del forward pass
for result in output_from_network:
    # Recorrer todos los objetos detectados
    for detected_objects in result:
        # detected_objects es un array con 85 elementos, donde los 4 primeros son las
        # posiciones del recuadro y el resto las probabilidades de la clase
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]

        # Checkpoint
        # Todo detected_objects tiene los 4 primeros elementos con bordes de la caja
        # y los 80 restantes con la clase a la que pertenece
        # print(detected_objects.shape)  #(85,)

        # Eliminar las predicciones debiles (que no sobrepasen un limite)
        if confidence_current > probability_minimum:
            # Escalar la caja al tamaño de la imagen
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            # Obtener la esquina superior izquierda que va a ser x_min e y_min
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            # Guardar los resultados en sus correspondientes listas
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)


"""
Non-maximum suppression
"""

# Con esta técnica, excluimos algunos cuadros delimitadores si sus
# correspondientes niveles de confianza son bajos o si existe otro
# cuadro delimitador para esta región con una confianza más alta.

# Es necesario asegurarse de que el tipo de dato de los cuadros
# delimitadores sea entero (int) y que el tipo de dato de los niveles
# de confianza sea de punto flotante (float).

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)


"""
Drawing bounding boxes and labels
"""

# Degining counter for detected objects
counter = 1

# Comprobar si al menos hay un objeto detectado
if len(results) > 0:
    # Recorrer los resultados
    for i in results.flatten():
        # Mostrar los etiquetas de los objetos detectados
        print("Object {0}: {1}".format(counter, labels[int(class_numbers[i])]))

        counter += 1

        # Definir el box
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        # Preparar los colores
        colour_box_current = colours[class_numbers[i]].tolist()

        # Dibujar la caja en la imagen original
        cv2.rectangle(
            image_BGR,
            (x_min, y_min),
            (x_min + box_width, y_min + box_height),
            colour_box_current,
            2,
        )

        # Preparar el texto con la etiqueta y la confianza necesaria
        text_box_current = "{}: {:.4f}".format(
            labels[int(class_numbers[i])], confidences[i]
        )

        # Poner el texto en la imagen original
        cv2.putText(
            image_BGR,
            text_box_current,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            colour_box_current,
            2,
        )


# Comparar cuantos objetos sin el non-maximum supppression
print()
print("Total de objetos detectados: ", len(bounding_boxes))
print("Numero de objetos tras el non-maximum: ", counter - 1)


# representar la imagen con los objetos detectados
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Detection", image_BGR)
cv2.waitKey(0)
cv2.destroyAllWindow("Detection")
