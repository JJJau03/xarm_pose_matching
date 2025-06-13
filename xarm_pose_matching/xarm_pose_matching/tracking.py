import cv2
from ultralytics import YOLO
import os
# Ruta al modelo YOLO entrenado
yolo_model_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
video_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/videos/RGB.mp4"
# Directorio que contiene las imágenes
image_dir = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/color"  # Ajusta esta ruta

# Cargar el modelo
model = YOLO(yolo_model_path)
class Lectura():
    def __init__(self):
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    def Video(self):
        # Abrir el video
        cap = cv2.VideoCapture(video_path)

        # Verificar que se haya abierto correctamente
        if not cap.isOpened():
            print("Error al abrir el video.")
            exit()

        # Procesar frame por frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video

            # Realizar inferencia con YOLO
            results = model(frame, show=False)
            boxes = results[0].boxes.xyxy
            width = frame.shape[1]
            for box in boxes:
                x1,y1,x2,y2 = box[:4]
                center_x = (x1+x2)/2
                  # Determinar posición relativa
                if center_x < width / 3:
                    print("Objeto a la IZQUIERDA")
                elif center_x > 2 * width / 3:
                    print("Objeto a la DERECHA")
                else:
                    print("Objeto al CENTRO")

            # Dibujar resultados directamente en el frame
            annotated_frame = results[0].plot()

            # Mostrar el frame con anotaciones
            cv2.imshow("YOLO Detection", annotated_frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

    def Imgs(self):
        # Obtener lista de imágenes
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(self.valid_extensions)]
        image_files.sort()  # Opcional: ordena alfabéticamente

        # Procesar cada imagen
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)

            # Leer imagen
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"No se pudo leer la imagen: {image_path}")
                continue

            # Realizar inferencia con YOLO
            results = model(frame, show=False)
            boxes = results[0].boxes.xyxy
            width = frame.shape[1]
            height = frame.shape[0]

            # Tamaño de cada zona
            cell_width = width / 5
            cell_height = height / 5
            annotated_frame = results[0].plot()
            # Dibujar la cuadrícula 5x5 en la imagen
            for i in range(1, 5):
                # Líneas verticales
                x = int(i * cell_width)
                cv2.line(annotated_frame, (x, 0), (x, height), color=(0, 255, 255), thickness=1)

                # Líneas horizontales
                y = int(i * cell_height)
                cv2.line(annotated_frame, (0, y), (width, y), color=(0, 255, 255), thickness=1)

            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Determinar zona en X y Y (0 a 4)
                zone_x = int(center_x // cell_width)
                zone_y = int(center_y // cell_height)

                # Asegurar que no se pase de 4 (por errores de redondeo flotante)
                zone_x = min(zone_x, 4)
                zone_y = min(zone_y, 4)

                print(f"{image_file}: Objeto en zona ({zone_x}, {zone_y})")
                
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), radius=5, color=(0, 0, 255), thickness=-1)

                # Dibujar resultados directamente en la imagen
            # Mostrar la imagen con anotaciones
            cv2.imshow("YOLO Detection", annotated_frame)

            # Esperar hasta que se presione una tecla (cerrar con 'q')
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    prueba = Lectura()
    prueba.Imgs()