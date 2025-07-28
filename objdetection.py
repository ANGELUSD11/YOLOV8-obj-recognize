import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") # 'n' = nano, es el más ligero y rápido
cap = cv2.VideoCapture(0)

try:
    width = int(input("Ingresa la anchura que quieras para el video: "))
    height = int(input("Ingresa la altura que quieras para el video: "))
except ValueError:
    print("Valor inválido, usando valores por defecto 640x480")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

#creo la ventana antes del bucle para asegurarme de que no solo use esa ventana
cv2.namedWindow("YOLOv8 Detección en Tiempo Real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detección en Tiempo Real", width, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    cv2.imshow("YOLOv8 Detección en Tiempo Real", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()