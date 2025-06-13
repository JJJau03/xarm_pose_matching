import open3d as o3d
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import trimesh
import cvzone
from PIL import Image, ImageDraw, ImageFont
import json

# Cargar el modelo YOLO (modelo de segmentación)
yolo_model_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
rgb_image_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/color/00002.jpg"
depth_image_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/depth/00002.png"
kinect_scan_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply"
model_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply"
video_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/videos/RGB.mp4"

with open("/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/intrinsic.json", 'r') as f:
    intrinsic_json = json.load(f)

# Convert flat list to 3x3 nested list
intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
intrinsic_matrix = [
    intrinsic_matrix_flat[0:3],
    intrinsic_matrix_flat[3:6],
    intrinsic_matrix_flat[6:9],
]

# Create PinholeCameraIntrinsic object
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsic_json['width'],
    intrinsic_json['height'],
    intrinsic_matrix[0][0],  # fx
    intrinsic_matrix[1][1],  # fy
    intrinsic_matrix[0][2],  # cx
    intrinsic_matrix[1][2],  # cy
)
# Cargar las imágenes
rgb_image = cv2.imread(rgb_image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# Cargar el modelo YOLO
model = YOLO(yolo_model_path)

target = o3d.io.read_point_cloud(kinect_scan_path) 
source = o3d.io.read_point_cloud(model_path)

# Ejecutar el modelo YOLO sin mostrar ni guardar automáticamente
results = model(rgb_image, show=False, save=False)

# Crear una copia de la imagen RGB para dibujarle encima
rgb_with_masks = rgb_image.copy()

# Crear figura con 3 subplots
fig, ax = plt.subplots(2, 3, figsize=(18, 6))

# Mostrar la imagen de profundidad original
ax[0,0].imshow(depth_image)
ax[0,0].set_title("Depth Image")
ax[0,0].axis('off')

# Iterar sobre los resultados y aplicar máscaras
for result in results:
    masks = result.masks.xy  # Lista de máscaras

    # Copias para procesar
    depth_with_mask = np.copy(depth_image)

    for mask in masks:
        mask = mask.astype(int)

        # Para superponer en profundidad
        mask_bin = np.zeros_like(depth_with_mask, dtype=np.uint8)
        cv2.fillPoly(mask_bin, [mask], 255)
        depth_with_mask = cv2.bitwise_and(depth_with_mask, depth_with_mask, mask=mask_bin)

        # Crear máscara binaria individual
        mask_bin = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask_bin, [mask], 255)
        # Conservar solo la parte dentro de la máscara
        masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_bin)
        # Combinar con la imagen final
        rgb_with_masks = cv2.bitwise_or(rgb_with_masks, masked_rgb)

hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
# color mask 
black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

# Filtering depth levels
min_depth = int(depth_with_mask.min())
max_depth = int(depth_with_mask.max())
lower_depth_bound = 1
upper_depth_bound = max_depth
depth_filtered = cv2.inRange(depth_with_mask, lower_depth_bound, upper_depth_bound)

#Combining masks
combined_mask = cv2.bitwise_and(black_mask, depth_filtered)
# Compute connected components
num_lablels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, 8, cv2.CV_32S)
# Find largest component
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
# Create a Mask of the largest component
cleaned_mask = np.where(labels == largest_label, 255, 0).astype('uint8')

result_rgb = cv2.bitwise_and(rgb_with_masks, rgb_with_masks, mask=cleaned_mask)
result_depth = cv2.bitwise_and(depth_with_mask, depth_with_mask, mask= cleaned_mask)

#RGB with depth image
o3d_color = o3d.geometry.Image(result_rgb)
o3d_depth = o3d.geometry.Image(result_depth)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
