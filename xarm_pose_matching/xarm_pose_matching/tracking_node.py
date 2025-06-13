import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
from matplotlib import pyplot as plt
from std_msgs.msg import Float32
from std_msgs.msg import String
from complete import PoseEstimator
from alignment import PointCloudVersion
import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
# Ruta al modelo YOLO
YOLO_MODEL_PATH = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/model/Segmentation_YOLO_v2.pt"
PLY_MODEL_PATH = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply"
class YoloImageSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_image_subscriber')
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image,'image_depth',self.depth_cb,10)
        #self.depth_sub = self.create_subscription(Image,'/kinect/depth/image_raw',self.depth_cb,10)
        self.pub = self.create_publisher(Image, '/video_processed', 10)
        self.pub_position = self.create_publisher(String, '/object_position', 10)  # Publicador posición
        self.mask_pub = self.create_publisher(Image, '/mask_depth', 10)
        self.pose_pub = self.create_publisher(Pose,'/object_pose',10)
        self.pub_min_depth = self.create_publisher(Float32, '/metrics/min_depth', 10)
        self.pub_max_depth = self.create_publisher(Float32, '/metrics/max_depth', 10)
        self.pub_rmse_wunsch = self.create_publisher(Float32, '/metrics/rmse_wunsch', 10)
        self.pub_rmse_ransac = self.create_publisher(Float32, '/metrics/rmse_ransac', 10)

        self.model = YOLO(YOLO_MODEL_PATH)
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.get_logger().info('Nodo de subscripción YOLO iniciado.')
        self.frame = None
        self.depth_frame = None
        self.pcd = o3d.io.read_point_cloud(PLY_MODEL_PATH)
        self.initial_transform = None
        #self.pcd.rotate(self.pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)), center=(0, 0, 0))

        self.subscription = self.create_subscription(Image, 'image_rgb', self.Img_cb, 10) 
        #self.subscription = self.create_subscription(Image, '/kinect/color/image_raw', self.Img_cb, 10) 

    def show_image_matplotlib(self,frame):
        # OpenCV usa BGR, matplotlib RGB, hay que convertir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')  # Oculta ejes
        plt.show()

    def depth_cb(self, msg):
        try:
            # Convertir el mensaje a una imagen OpenCV (profundidad normalmente es 16UC1 o 32FC1)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Opcional: Normalizar para visualización (si lo quieres mostrar o guardar)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)

            # Puedes guardar o usar esta imagen en otras funciones del nodo
            self.depth_frame = depth_image  # Guardarla para uso posterior si quieres

        except Exception as e:
            self.get_logger().error(f"Error al convertir imagen de profundidad: {str(e)}")

    def tracking(self,frame):
        if frame is None:
            print(f"No se pudo leer la imagen:")
        height, width = frame.shape[:2]
        # Realizar inferencia con YOLO
        results = self.model(frame, show=False)
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
        positions = []
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
            positions.append(f"({zone_x},{zone_y})")
            #print(f"Objeto en zona ({zone_x}, {zone_y})")
            
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), radius=5, color=(0, 0, 255), thickness=-1)
        msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.pub.publish(msg)
        if positions:
            pos_msg = String()
            pos_msg.data = ";".join(positions)
            self.pub_position.publish(pos_msg)
        else:
            # Publicar vacío o algo si no hay objetos detectados
            pos_msg = String()
            pos_msg.data = "No objects"
            self.pub_position.publish(pos_msg)
        
    def Img_cb(self,msg):
        #Librerias
        processor = PointCloudVersion()
        preprocessor = PoseEstimator()
        pose_msg = Pose()

        # Leer imagen
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #Procesamiento
        self.tracking(frame)
        min_depth, max_depth, target, mask_depth = preprocessor.main(frame,self.depth_frame)
        rmse_wunsch,rmse_ransac,position = processor.main(target,self.pcd)
        msg_depth = self.bridge.cv2_to_imgmsg(mask_depth, encoding='16UC1')  # o 'mono8', '32FC1', según el tipo
        T = position
        pose_msg.position.x = float(T[0, 3])
        pose_msg.position.y = float(T[1, 3])
        pose_msg.position.z = float(T[2, 3])

        # Rotación: matriz 3x3 a cuaternión
        rot = R.from_matrix(T[:3, :3])
        quat = rot.as_quat()  # Formato: [x, y, z, w]

        pose_msg.orientation.x = float(quat[0])
        pose_msg.orientation.y = float(quat[1])
        pose_msg.orientation.z = float(quat[2])
        pose_msg.orientation.w = float(quat[3])

        # Publicar en el topic
        self.pose_pub.publish(pose_msg)
        #Publishers
        self.mask_pub.publish(msg_depth)
        
        min_msg = Float32()
        min_msg.data = float(min_depth)
        self.pub_min_depth.publish(min_msg)

        max_msg = Float32()
        max_msg.data = float(max_depth)
        self.pub_max_depth.publish(max_msg)

        rmse_w_msg = Float32()
        rmse_w_msg.data = float(rmse_wunsch)
        self.pub_rmse_wunsch.publish(rmse_w_msg)

        rmse_r_msg = Float32()
        rmse_r_msg.data = float(rmse_ransac)
        self.pub_rmse_ransac.publish(rmse_r_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Nodo interrumpido por el usuario.")
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
