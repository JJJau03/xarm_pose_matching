import open3d as o3d
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import trimesh
import cvzone
from PIL import Image, ImageDraw, ImageFont
import json

yolo_model_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
rgb_image_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/color/00002.jpg"
depth_image_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/depth/00002.png"
kinect_scan_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/rgb_image_V2.ply"
model_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply"

class PoseEstimator():
    def __init__(self):
        with open("/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/intrinsic.json", 'r') as f:
            intrinsic_json = json.load(f)
        intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
        intrinsic_matrix = [
            intrinsic_matrix_flat[0:3],
            intrinsic_matrix_flat[3:6],
            intrinsic_matrix_flat[6:9],
        ]

        # Create PinholeCameraIntrinsic object
        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsic_json['width'],
            intrinsic_json['height'],
            intrinsic_matrix[0][0],  # fx
            intrinsic_matrix[1][1],  # fy
            intrinsic_matrix[0][2],  # cx
            intrinsic_matrix[1][2],  # cy
        )
        self.model = YOLO(yolo_model_path)
        self.rgb_image = None
        self.depth_image = None
        self.depth_with_mask = None
        self.depth_filtered = None
        self.cleaned_mask = None
        self.result_rgb = None
        self.result_depth = None
        self.centroid = None
        self.penduncle_point = None
        self.pcd = None

    def processing(self):
        rgb_image = cv2.imread(rgb_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        results = self.model(rgb_image, show=False, save=False)
        rgb_with_masks = rgb_image.copy()
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
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.depth_with_mask = depth_with_mask
        self.depth_filtered = depth_filtered
        self.cleaned_mask = cleaned_mask
        self.result_rgb = result_rgb
        self.result_depth = result_depth

    def ply_generator(self):
        o3d_color = o3d.geometry.Image(self.result_rgb)
        o3d_depth = o3d.geometry.Image(self.result_depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsic)
        pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        self.pcd = pcd
        #o3d.io.write_point_cloud("/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/rgb_image_V2.ply", pcd)

    def plot(self):
        # Crear figura con 3 subplots
        fig, ax = plt.subplots(2, 3, figsize=(18, 6))
        # Mostrar la imagen de profundidad original
        ax[0,0].imshow(self.depth_image)
        ax[0,0].set_title("Depth Image")
        ax[0,0].axis('off')
        ax[0,1].imshow(self.depth_with_mask)
        ax[0,1].set_title("Depth with Mask")
        ax[0,1].axis('off')

        # Mostrar imagen de profundidad filtrada
        ax[0,2].imshow(self.depth_filtered, cmap='gray')
        ax[0,2].set_title("Depth Filtered")
        ax[0,2].axis('off')

        ax[1,0].imshow(self.cleaned_mask, cmap='gray')
        ax[1,0].set_title("Cleaned Mask")
        ax[1,0].axis("off")

        ax[1,1].imshow(cv2.cvtColor(self.result_rgb, cv2.COLOR_BGR2RGB))
        ax[1,1].set_title("Depth Result Mask")
        ax[1,1].axis("off")

        ax[1,2].hist(self.depth_image.ravel(), bins=100, color='blue', alpha=0.7)
        ax[1,2].set_title("Depth Histogram")
        ax[1,2].set_xlabel("Depth Value")
        ax[1,2].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        o3d.visualization.draw_geometries([self.pcd])
    
    def estimator(self):
        #Line from center to peduncle
        model = o3d.io.read_point_cloud(model_path) 
        direction, centroid, peduncle_point = self.compute_line(model)
        direction_scan, centroid_scan, peduncle_point_scan = self.compute_line(self.pcd)
        #Translated Models
        self.pcd = self.coordinate_frame(self.pcd,peduncle_point_scan)
        #model = self.coordinate_frame(model,peduncle_point_scan)
        
        #Rotation
        initial_direction = direction / np.linalg.norm(direction)
        target_direction = direction_scan / np.linalg.norm(direction_scan)

        # Compute the rotation
        axis, angle = self.compute_rotation(initial_direction, target_direction)
        R = self.axis_angle_to_rotation_matrix(axis, angle)
        source_points = np.asarray(model.points)

        # Apply rotation to 3D model
        rotated_points_np = self.apply_rotation(source_points, R)
        rotated_pcd = o3d.geometry.PointCloud()
        rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points_np)
        o3d.visualization.draw_geometries([rotated_pcd, self.pcd])

        #Rotate along Z-axis
        angle_180_degrees = (np.pi)  # 180° in radians
        R_180 = self.rotation_matrix_around_z(angle_180_degrees)
        flipped_points_np = self.apply_rotation(rotated_points_np, R_180)
        flipped_pcd = o3d.geometry.PointCloud()
        flipped_pcd.points = o3d.utility.Vector3dVector(flipped_points_np)
        o3d.visualization.draw_geometries([flipped_pcd, self.pcd])

        #Scaling the model
        target_dimensions = self.get_dimensions(self.pcd)
        scaled_pcd2 = self.scale_point_cloud(flipped_pcd, target_dimensions)

        # We paint the point clouds to be able to distinguish between them
        self.pcd.paint_uniform_color([1, 0, 0])  # Paint the first point cloud red
        scaled_pcd2.paint_uniform_color([0, 1, 0])  # Paint the scaled point cloud green
        o3d.visualization.draw_geometries([self.pcd, scaled_pcd2])
                # === ICP Final ===
        threshold = 0.01  # ajusta este valor si es necesario
        print("Ejecutando ICP...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            scaled_pcd2,  # source: el modelo ajustado (escalado + rotado)
            self.pcd,     # target: nube segmentada de la escena
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        print("Transformación ICP:")
        print(reg_p2p.transformation)

        # Aplicar transformación estimada al modelo
        final_model = scaled_pcd2.transform(reg_p2p.transformation)

        # Visualización final
        self.pcd.paint_uniform_color([1, 0, 0])         # escena en rojo
        final_model.paint_uniform_color([0, 1, 0])      # modelo en verde
        o3d.visualization.draw_geometries([self.pcd, final_model])

    def main(self):
        self.processing()
        self.ply_generator()
        self.plot()
        self.estimator()
    
    def compute_centroid(self,model): 
        return np.mean(model.points, axis=0)
    #Resources
    def identify_peduncle_point(self,model):
        coordinates = np.asarray(model.points)[:,2]
        threshold = np.percentile(coordinates,98)
        top_points = np.asarray(model.points)[coordinates>threshold]
        return np.mean(top_points,axis=0)
    
    def compute_line(self,model):
        #base
        centroid = self.compute_centroid(model)
        peduncle_point= self.identify_peduncle_point(model)
        direction_vector = peduncle_point - centroid
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)
        #Aumentado line from center to peduncle
        line_points = [centroid, peduncle_point]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector([[0,1]])
        o3d.visualization.draw_geometries([model,line_set])
        return normalized_vector, centroid, peduncle_point

    def coordinate_frame(self, model, peduncle_point):
        #Para colocar en el centro de la figura se agregan las siguientes lineas
        centroid = np.mean(np.asarray(model.points), axis=0)
        translated_points = np.asarray(model.points) - centroid
        model.points = o3d.utility.Vector3dVector(translated_points)
        #Coordinate Frame
        centroid_pcd = o3d.geometry.PointCloud()
        centroid_pcd.points = o3d.utility.Vector3dVector([centroid])
        centroid_color = [1,0,0]
        centroid_pcd.colors = o3d.utility.Vector3dVector([centroid_color])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])
        o3d.visualization.draw_geometries([model,centroid_pcd,coord_frame])
        #Translation
        line_points = [centroid, peduncle_point]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        o3d.visualization.draw_geometries([model, line_set])
        return model

              
    def compute_rotation(self,v1,v2):
         # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute rotation axis
        rotation_axis = np.cross(v1, v2)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Compute rotation angle
        cos_angle = np.dot(v1, v2)
        rotation_angle = np.arccos(cos_angle)
        
        return rotation_axis, rotation_angle
    
    def axis_angle_to_rotation_matrix(self, axis, angle):
        # Using the Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R
    
    def apply_rotation(self,pcd, rotation_matrix):
        return np.dot(pcd, rotation_matrix.T)  # .T is for transpose   
    
    def rotation_matrix_around_x(self,angle_rad):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    
    def rotation_matrix_around_y(self,angle_rad) :
        return np.array([
            [np.cos(angle_rad),0,np.sin(angle_rad)],
            [0,1,0],
            [-np.sin(angle_rad),0,np.cos(angle_rad)]
        ])
    
    def rotation_matrix_around_z(self,angle_rad):
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    def get_dimensions(self, model):
        bounding_box = model.get_axis_aligned_bounding_box()
        return bounding_box.get_extent()
    
    def scale_point_cloud(self,source_pcd, target_dimensions, source_dimensions=None):
        if source_dimensions is None:
            source_dimensions = self.get_dimensions(source_pcd)
        
        scale_factors = [
            target_dimensions[i] / source_dimensions[i]
            for i in range(3)
        ]

        scaled_points = [
            [scale_factors[j] * pt[j] for j in range(3)]
            for pt in source_pcd.points
        ]

        scaled_pcd = o3d.geometry.PointCloud()
        scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
        return scaled_pcd
    
if __name__=="__main__":#
    scene1 = PoseEstimator()
    scene1.main()


#direction, centroid, peduncle_point = compute_line(self.pcd)
#direction_scan, centroid_scan, peduncle_point_scan = compute_line_scan(model)
#direction_scan, centroid_scan, peduncle_point_scan = compute_line_scan(translated_kinect_scan)