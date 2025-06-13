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

            # Dibujar resultados directamente en la imagen
            annotated_frame = results[0].plot()

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