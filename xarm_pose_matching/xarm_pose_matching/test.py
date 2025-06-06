# import open3d as o3d
# import numpy as np
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import cv2
# import trimesh
# import cvzone
# from PIL import Image, ImageDraw,ImageFont

# # Cargar el modelo YOLO (modelo de segmentación)
# yolo_model_path = "/home/jjj/xarm_kinect_ros2/src/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
# rgb_image_path = "/home/jjj/Open3D/examples/python/reconstruction_system/sensors/data/color/00002.jpg"
# depth_image_path = "/home/jjj/Open3D/examples/python/reconstruction_system/sensors/data/depth/00002.png"

# # Cargar las imágenes
# rgb_image = cv2.imread(rgb_image_path)
# depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# # Cargar el modelo YOLO
# model = YOLO(yolo_model_path)

# # Realizar la segmentación (predicción)
# results = model(rgb_image, show=True, save=True)

# for result in results:
#     # Get the height and width of the original image
#     height, width = result.orig_img.shape[:2]

#     # Create the background
#     background = np.ones((height, width, 3), dtype=np.uint8) * 255
    
#     # Get all predicted masks
#     masks = result.masks.xy

#     # Get the original image
#     orig_img = result.orig_img

#     for mask in masks:
#         mask = mask.astype(int)

#         # Create a mask image
#         mask_img = np.zeros_like(orig_img)

#         # Fill the contour of the mask image in white
#         cv2.fillPoly(mask_img, [mask], (255, 255, 255))

#         # Extract the object from the original image using the mask
#         masked_object = cv2.bitwise_and(orig_img, mask_img)

#         # Copy the masked object to the background image
#         background[mask_img == 255] = masked_object[mask_img == 255]

# # Display the result
# cv2.imshow('Segmented objects', background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save the result
# cv2.imwrite('segmented_objects.jpg', background)
import open3d as o3d
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import trimesh
import cvzone
from PIL import Image, ImageDraw, ImageFont

# Cargar el modelo YOLO (modelo de segmentación)
yolo_model_path = "/home/jjj/xarm_kinect_ros2/src/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
rgb_image_path = "/home/jjj/Open3D/examples/python/reconstruction_system/sensors/data/color/00002.jpg"
depth_image_path = "/home/jjj/Open3D/examples/python/reconstruction_system/sensors/data/depth/00002.png"

# Cargar las imágenes
rgb_image = cv2.imread(rgb_image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# Cargar el modelo YOLO
model = YOLO(yolo_model_path)

# Realizar la segmentación (predicción)
results = model(rgb_image, show=True, save=True)

# Crear la figura para mostrar ambas imágenes
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Muestra la imagen de profundidad en la primera columna
ax[0].imshow(depth_image, cmap='gray')
ax[0].set_title("Depth Image")
ax[0].axis('off')  # Ocultar ejes

# Iterar sobre los resultados de la segmentación y aplicar la máscara sobre la imagen de profundidad
for result in results:
    # Obtener las máscaras predichas
    masks = result.masks.xy

    # Crear una copia de la imagen de profundidad para superponer la máscara
    depth_with_mask = np.copy(depth_image)

    for mask in masks:
        mask = mask.astype(int)

        # Crear una máscara binaria (uint8 en lugar de uint16 para la operación bitwise_and)
        mask_bin = np.zeros_like(depth_with_mask, dtype=np.uint8)  # Cambiar a uint8
        cv2.fillPoly(mask_bin, [mask], 255)  # Usar 255 para la máscara binaria

        # Aplicar la máscara sobre la imagen de profundidad (la imagen de profundidad sigue siendo uint16)
        depth_with_mask = cv2.bitwise_and(depth_with_mask, depth_with_mask, mask=mask_bin)

    # Muestra la imagen de profundidad con la máscara aplicada en la segunda columna
    ax[1].imshow(depth_with_mask, cmap='gray')
    ax[1].set_title("Depth with Mask")
    ax[1].axis('off')  # Ocultar ejes

# Mostrar la figura con ambas imágenes
plt.tight_layout()
plt.show()

# Guardar la imagen resultante
cv2.imwrite('depth_with_mask.jpg', depth_with_mask)
