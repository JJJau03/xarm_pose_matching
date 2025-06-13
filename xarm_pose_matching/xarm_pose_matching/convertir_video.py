# import cv2
# import os

# # Directorio donde están las imágenes
# image_folder = '/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/color'

# # Nombre del archivo de video de salida
# video_name = '/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/videos/RGB.mp4'

# # Obtener lista de archivos en el directorio
# images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
# images.sort()  # Asegúrate de que estén en el orden correcto

# # Leer la primera imagen para obtener tamaño del video
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# # Definir el codec y crear el objeto VideoWriter
# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 24, (width, height))

# # Agregar cada imagen al video
# for image in images:
#     frame = cv2.imread(os.path.join(image_folder, image))
#     video.write(frame)

# video.release()
# print("✅ Video creado exitosamente:", video_name)

import cv2
import os
import numpy as np

image_folder = '/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/depth'
video_name = '/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/videos/Depth.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

# Leer la primera imagen para tamaño
depth_image = cv2.imread(os.path.join(image_folder, images[0]), cv2.IMREAD_UNCHANGED)
height, width = depth_image.shape

# Crear el video (3 canales porque convertimos la profundidad a color)
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

for image in images:
    depth_raw = cv2.imread(os.path.join(image_folder, image), cv2.IMREAD_UNCHANGED)

    # Normalizar profundidad para que esté entre 0-255 (8-bit)
    depth_normalized = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)

    # Aplicar un colormap (opcional pero útil para visualización)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    video.write(depth_colored)

video.release()
print("✅ Video de profundidad creado exitosamente:", video_name)
