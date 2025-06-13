# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# class VideoPublisher(Node):
#     def __init__(self):
#         super().__init__('video_publisher')
#         self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
#         self.bridge = CvBridge()
#         self.cap = cv2.VideoCapture('/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/videos/2025-06-03-21-19-19.mkv')  # Cambia esto a la ruta real del video
#         self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30 FPS

#     def timer_callback(self): 
#         ret, frame = self.cap.read()
#         if not ret:
#             self.get_logger().info('Fin del video o error de lectura.')
#             self.cap.release()
#             self.destroy_timer(self.timer)
#             return

#         msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
#         self.publisher_.publish(msg)
#         self.get_logger().info('Publicando un cuadro de video.')

# def main(args=None):
#     rclpy.init(args=args)
#     node = VideoPublisher()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class RGBDImagePublisher(Node):
    def __init__(self):
        super().__init__('rgbd_image_publisher')

        self.rgb_publisher = self.create_publisher(Image, 'image_rgb', 10)
        self.depth_publisher = self.create_publisher(Image, 'image_depth', 10)
        self.bridge = CvBridge()

        # Directorios
        self.rgb_dir = '/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/color'
        self.depth_dir = '/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/depth'

        # Extensiones válidas
        self.valid_extensions = ('.png', '.jpg', '.jpeg')

        # Archivos RGB
        self.image_filenames = sorted([
            f for f in os.listdir(self.rgb_dir)
            if f.lower().endswith(self.valid_extensions)
        ])

        if not self.image_filenames:
            self.get_logger().error('No se encontraron imágenes RGB.')
            return

        self.index = 0
        self.timer = self.create_timer(0.5, self.timer_callback)  # 2 Hz

    def timer_callback(self):
        if self.index >= len(self.image_filenames):
            self.get_logger().info('Se han publicado todas las imágenes.')
            self.destroy_timer(self.timer)
            return

        rgb_filename = self.image_filenames[self.index]
        rgb_path = os.path.join(self.rgb_dir, rgb_filename)

        # Obtener base del nombre (sin extensión)
        base_name = os.path.splitext(rgb_filename)[0]
        depth_path = os.path.join(self.depth_dir, base_name + ".png")

        # Leer imágenes
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_image is None:
            self.get_logger().warn(f'No se pudo leer la imagen RGB: {rgb_filename}')
        else:
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            rgb_msg.header.frame_id = "camera_rgb_optical_frame"
            self.rgb_publisher.publish(rgb_msg)

        if depth_image is None:
            self.get_logger().warn(f'No se pudo leer la imagen de profundidad: {os.path.basename(depth_path)}')
        else:
            # Codificación automática según tipo
            if depth_image.dtype == 'uint16':
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
            elif depth_image.dtype == 'float32':
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            else:
                self.get_logger().warn(f'Formato de imagen de profundidad no compatible: {depth_image.dtype}')
                self.index += 1
                return

            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.depth_publisher.publish(depth_msg)

        self.get_logger().info(f'Publicando imagen RGB-D: {rgb_filename}')
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = RGBDImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
