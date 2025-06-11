import os
import cv2
import json
import numpy as np
import open3d as o3d
from ultralytics import YOLO

# === Paths ===
base_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data"
write_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/ply_scenes"
rgb_dir = os.path.join(base_path, "color")
depth_dir = os.path.join(base_path, "depth")
output_dir = os.path.join(write_path, "output_ply")
model_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
intrinsic_path = os.path.join(base_path, "intrinsic.json")

# === Cargar modelo YOLO ===
model = YOLO(model_path)

# === Cargar intrínsecos de la cámara ===
with open(intrinsic_path, 'r') as f:
    intrinsic_json = json.load(f)

intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
intrinsic_matrix = [
    intrinsic_matrix_flat[0:3],
    intrinsic_matrix_flat[3:6],
    intrinsic_matrix_flat[6:9],
]

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsic_json['width'],
    intrinsic_json['height'],
    intrinsic_matrix[0][0],  # fx
    intrinsic_matrix[1][1],  # fy
    intrinsic_matrix[0][2],  # cx
    intrinsic_matrix[1][2],  # cy
)

# === Crear carpeta de salida si no existe ===
os.makedirs(output_dir, exist_ok=True)

# === Obtener todas las imágenes RGB ===
rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])

for rgb_file in rgb_files:
    scene_name = os.path.splitext(rgb_file)[0]
    rgb_path = os.path.join(rgb_dir, rgb_file)
    depth_path = os.path.join(depth_dir, f"{scene_name}.png")
    output_ply_path = os.path.join(output_dir, f"{scene_name}.ply")

    # Cargar imágenes
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Ejecutar modelo YOLO
    results = model(rgb_image, show=False, save=False)

    depth_with_mask = np.copy(depth_image)
    rgb_with_masks = rgb_image.copy()

    for result in results:
        masks = result.masks.xy

        for mask in masks:
            mask = mask.astype(int)
            mask_bin = np.zeros_like(depth_with_mask, dtype=np.uint8)
            cv2.fillPoly(mask_bin, [mask], 255)
            depth_with_mask = cv2.bitwise_and(depth_with_mask, depth_with_mask, mask=mask_bin)

            mask_bin_rgb = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask_bin_rgb, [mask], 255)
            masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_bin_rgb)
            rgb_with_masks = cv2.bitwise_or(rgb_with_masks, masked_rgb)

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 50]))

    lower_depth_bound = 1
    upper_depth_bound = int(depth_with_mask.max())
    depth_filtered = cv2.inRange(depth_with_mask, lower_depth_bound, upper_depth_bound)

    combined_mask = cv2.bitwise_and(black_mask, depth_filtered)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, 8, cv2.CV_32S)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = np.where(labels == largest_label, 255, 0).astype('uint8')

    result_rgb = cv2.bitwise_and(rgb_with_masks, rgb_with_masks, mask=cleaned_mask)
    result_depth = cv2.bitwise_and(depth_with_mask, depth_with_mask, mask=cleaned_mask)

    o3d_color = o3d.geometry.Image(result_rgb)
    o3d_depth = o3d.geometry.Image(result_depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

    # Guardar archivo PLY
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"Guardado: {output_ply_path}")
