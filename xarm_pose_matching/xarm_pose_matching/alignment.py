import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class TriangleMeshVersion():
    def __init__(self):
        pass

    def draw_registration_result(self,source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    def preprocess_point_cloud(self,pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh


    def execute_global_registration(self,source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result


    def main(self):
        voxel_size = 0.01
        
        print(":: Load two mesh.")
        target_mesh = o3d.io.read_triangle_mesh('bunny.ply')
        source_mesh = copy.deepcopy(target_mesh)
        source_mesh.rotate(source_mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)), center=(0, 0, 0))
        source_mesh.translate((0, 0.05, 0))
        self.draw_registration_result(target_mesh, source_mesh, np.identity(4))

        print(":: Sample mesh to point cloud")
        target = target_mesh.sample_points_uniformly(1000)
        source = source_mesh.sample_points_uniformly(1000)
        self.draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        result_ransac = self.execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        print(result_ransac)
        self.draw_registration_result(source_down, target_down, result_ransac.transformation)
        self.draw_registration_result(source_mesh, target_mesh, result_ransac.transformation)

class PointCloudVersion():
    def __init__(self):
        self.source_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply"
        self.target_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/rgb_image_V2.ply"
        self.ply2_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/ply_scenes/output_ply/00010.ply"
        self.ply3_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/ply_scenes/output_ply/00067.ply"

    def draw_registration_result(self,source, target, transformation):   
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])     # naranja
        target_temp.paint_uniform_color([0, 0.651, 0.929]) # azul
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def print_matrix(self,transformation,option):
        # Extraer rotación y traslación
        rotation_matrix = transformation[:3, :3]
        translation_vector = transformation[:3, 3]
        rot = R.from_matrix(rotation_matrix)
        euler_angles = rot.as_euler('xyz', degrees=True)  # puedes cambiar el orden a 'zyx', etc.
        # Mostrar los valores
        if option ==0:
            print("\nPosición Estimada RANSAC")
        elif option == 1:
            print("\nPosición Estimada Algoritmo de Wunsch")
        print("\nMatriz de rotación:")
        print(rotation_matrix)
        print("\nVector de traslación:")
        print(translation_vector)
        print("\nÁngulos de Euler (grados):", euler_angles)


    def preprocess_point_cloud(self,pcd, voxel_size):
        #print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        #print(":: Estimate normals with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    #ineficiente
    def calculate_correspondence_accuracy(self,result, source_down, target_down):
        num_corr = len(result.correspondence_set)
        total_points = min(len(source_down.points), len(target_down.points))
        accuracy = (num_corr / total_points) * 100  # como porcentaje
        return accuracy
    
    def calculate_rmse(self,result, source_down, target_down):
        src = np.asarray(source_down.points)
        tgt = np.asarray(target_down.points)
        transformation = result.transformation
        transformed_src = (transformation[:3,:3] @ src.T).T + transformation[:3,3]

        correspondences = np.array(result.correspondence_set)
        src_corr = transformed_src[correspondences[:, 0]]
        tgt_corr = tgt[correspondences[:, 1]]

        errors = np.linalg.norm(src_corr - tgt_corr, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        return rmse

    def execute_global_registration(self,source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        #print(":: RANSAC registration on downsampled point clouds.")
        #print("   Using distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result
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
    def load_and_scale_ply(self, target, source):
        # Obtener las dimensiones (bounding boxes)
        source_bbox = source.get_axis_aligned_bounding_box()
        target_bbox = target.get_axis_aligned_bounding_box()

        # Calcular escalas
        source_extent = source_bbox.get_extent()
        target_extent = target_bbox.get_extent()
        # Escalar usando la proporción de la diagonal de la caja
        scale_factor = np.linalg.norm(target_extent) / np.linalg.norm(source_extent)

        # Centrar el source en el origen antes de escalar
        source_centered = source.translate(-source.get_center())
        source_scaled = source_centered.scale(scale_factor, center=(0, 0, 0))
        
        # Reubicarlo en el centro del target (opcional)
        source_scaled.translate(target.get_center())
        return source_scaled
    
    def load_and_scale_ply2(self, target, source):
        source_bbox = source.get_axis_aligned_bounding_box()
        target_bbox = target.get_axis_aligned_bounding_box()

        source_extent = source_bbox.get_extent()
        target_extent = target_bbox.get_extent()

        scale_factors = target_extent / source_extent  # Escala por eje X, Y, Z

        # Centrar el source antes de escalar
        source_center = source.get_center()
        source.translate(-source_center)

        # Aplicar escala por eje (no uniforme)
        source_points = np.asarray(source.points)
        scaled_points = source_points * scale_factors
        source.points = o3d.utility.Vector3dVector(scaled_points)

        # Volver a trasladar al centro original del target
        source.translate(target.get_center())

        return source

    
    
    def igualar_puntos(self,pcd1, pcd2):
        n1 = len(pcd1.points)
        n2 = len(pcd2.points)

        target_n = min(n1, n2)  # o usa max() si quieres más puntos

        pcd1_down = pcd1.farthest_point_down_sample(target_n) if n1 > target_n else pcd1
        pcd2_down = pcd2.farthest_point_down_sample(target_n) if n2 > target_n else pcd2

        return pcd1_down, pcd2_down
    

     # ============== WUNSCH ==================== #
    def refine_with_svd(self,source_points, target_points):
        """
        Refina la transformación usando SVD como en el código Wunsch
        """
        # Calcular centros de masa
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)
        
        # Centrar los puntos
        centered_source = source_points - centroid_source
        centered_target = target_points - centroid_target
        
        # Matriz de covarianza (similar a Qxy en Wunsch)
        H = centered_source.T @ centered_target
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Calcular rotación
        R = Vt.T @ U.T
        
        # Manejar reflexiones
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        # Calcular traslación
        t = centroid_target - R @ centroid_source
        
        # Crear matriz de transformación 4x4
        transformation = np.identity(4)
        transformation[:3,:3] = R
        transformation[:3,3] = t
        
        return transformation
    
    def execute_global_registration_Wunsch(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        # Paso 1: Registro inicial con RANSAC
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        if not result.correspondence_set:
            return result
        
        # Paso 2: Refinamiento iterativo con SVD ponderado
        source_points = np.asarray(source_down.points)
        target_points = np.asarray(target_down.points)
        correspondences = np.array(result.correspondence_set)
        
        for _ in range(3):  # 3 iteraciones de refinamiento
            # Obtener puntos correspondientes
            src_pts = source_points[correspondences[:,0]]
            tgt_pts = target_points[correspondences[:,1]]
            
            # Calcular pesos
            weights = self.compute_weights_Wunsch(src_pts, tgt_pts, result.transformation)
            
            # Refinar con SVD ponderado
            refined_trans = self.weighted_svd_refinement_Wunsch(src_pts, tgt_pts, weights)
            result.transformation = refined_trans
            
            # Actualizar correspondencias basado en la nueva transformación
            transformed_src = (refined_trans[:3,:3] @ source_points.T).T + refined_trans[:3,3]
            dists = np.linalg.norm(transformed_src[:,np.newaxis] - target_points, axis=2)
            correspondences = np.stack([np.arange(len(source_points)), np.argmin(dists, axis=1)], axis=1)
        
        return result
    
    def compute_weights_Wunsch(self,source_points, target_points, transformation, sigma=0.1):
        """
        Calcula pesos para cada correspondencia basado en el error
        """
        transformed_source = (transformation[:3,:3] @ source_points.T).T + transformation[:3,3]
        errors = np.linalg.norm(transformed_source - target_points, axis=1)
        weights = np.exp(-errors**2/(2*sigma**2))
        return weights

    def weighted_svd_refinement_Wunsch(self,source_points, target_points, weights):
        """
        Versión ponderada del refinamiento SVD
        """
        weights = weights.reshape(-1,1)
        centroid_source = np.sum(weights*source_points, axis=0) / np.sum(weights)
        centroid_target = np.sum(weights*target_points, axis=0) / np.sum(weights)
        
        centered_source = source_points - centroid_source
        centered_target = target_points - centroid_target
        
        # Matriz de covarianza ponderada
        H = (weights * centered_source).T @ centered_target
        
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        t = centroid_target - R @ centroid_source
        
        transformation = np.identity(4)
        transformation[:3,:3] = R
        transformation[:3,3] = t
        
        return transformation
    
    def main(self,target,source):
        voxel_size = 0.01

        #print(":: Load target and source point clouds")
        # target = o3d.io.read_point_cloud(self.target_path)
        # source = o3d.io.read_point_cloud(self.source_path)
        # Simular transformación en el source
        #source.rotate(source.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)), center=(0, 0, 0))
        #source.translate((0, 0.05, 0))
        # ======= source and target =================
        #self.draw_registration_result(source, target, np.identity(4))

        #target_dimensions = self.get_dimensions(target)
        #scaled_source = self.scale_point_cloud(source, target_dimensions)
        scaled_source = self.load_and_scale_ply(target,source)
        #o3d.visualization.draw_geometries([target,scaled_source])

        # Preprocesamiento: downsample + normales + FPFH
        source_down, source_fpfh = self.preprocess_point_cloud(scaled_source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)

        # Registro global con RANSAC 
        result_ransac = self.execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        
        result_ransac_Wunsch = self.execute_global_registration_Wunsch(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        rmse_ransac = self.calculate_rmse(result_ransac, source_down, target_down)
        rmse_wunsch = self.calculate_rmse(result_ransac_Wunsch, source_down, target_down)

        # print(result_ransac)
        # self.print_matrix(result_ransac.transformation,0)
        # # print(f"RANSAC - Porcentaje de correspondencias válidas: {accuracy_ransac:.2f}%")
        # # accuracy_ransac = self.calculate_correspondence_accuracy(result_ransac, source_down, target_down)
        # print(f"RANSAC - RMSE de correspondencias: {rmse_ransac:.4f}")
        # self.draw_registration_result(source_down, target_down, result_ransac.transformation)
        # self.draw_registration_result(scaled_source, target, result_ransac.transformation)
        # self.print_matrix(result_ransac.transformation,1)
        # # accuracy_wunsch = self.calculate_correspondence_accuracy(result_ransac_Wunsch, source_down, target_down)
        # # print(f"Wunsch - Porcentaje de correspondencias válidas: {accuracy_wunsch:.2f}%")
        # print(f"Wunsch - RMSE de correspondencias: {rmse_wunsch:.4f}")
        # self.draw_registration_result(source_down, target_down, result_ransac_Wunsch.transformation)
        # self.draw_registration_result(scaled_source, target, result_ransac_Wunsch.transformation)
        return rmse_wunsch,rmse_ransac,result_ransac_Wunsch.transformation # Return for ros2 node
        #return rmse_wunsch,rmse_ransac, scaled_source  # Return for final in PoseEstimator python code

        
        # ## ======= ply2 and target =================
        # ply_2 = o3d.io.read_point_cloud(self.ply2_path)
        # ply_3 = o3d.io.read_point_cloud(self.ply3_path)
        # ply_2.translate((1, 0.0, 0.01))
        # ply_3.translate((-0.3, 0.7, 1.8))
        
        # self.draw_registration_result(source, ply_2, np.identity(4))

        # target_dimensions = self.get_dimensions(target)
        # #scaled_source = self.scale_point_cloud(source, target_dimensions)
        # scaled_source = self.load_and_scale_ply(ply_2,source)
        # o3d.visualization.draw_geometries([ply_2,scaled_source])

        # # Preprocesamiento: downsample + normales + FPFH
        # source_down, source_fpfh = self.preprocess_point_cloud(scaled_source, voxel_size)
        # target_down, target_fpfh = self.preprocess_point_cloud(ply_2, voxel_size)

        # # Registro global con RANSAC 
        # result_ransac = self.execute_global_registration(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        
        # result_ransac_Wunsch = self.execute_global_registration_Wunsch(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        # print(result_ransac)
        # self.print_matrix(result_ransac.transformation,0)
        # self.draw_registration_result(source_down, target_down, result_ransac.transformation)
        # self.draw_registration_result(scaled_source, ply_2, result_ransac.transformation)
        # self.print_matrix(result_ransac.transformation,1)
        # self.draw_registration_result(source_down, target_down, result_ransac_Wunsch.transformation)
        # self.draw_registration_result(scaled_source, ply_2, result_ransac_Wunsch.transformation)
        # ## ======= ply3 and target =================
        # self.draw_registration_result(source, ply_3, np.identity(4))

        # target_dimensions = self.get_dimensions(ply_3)
        # #scaled_source = self.scale_point_cloud(source, target_dimensions)
        # scaled_source = self.load_and_scale_ply(ply_3,source)
        # o3d.visualization.draw_geometries([ply_3,scaled_source])

        # # Preprocesamiento: downsample + normales + FPFH
        # source_down, source_fpfh = self.preprocess_point_cloud(scaled_source, voxel_size)
        # target_down, target_fpfh = self.preprocess_point_cloud(ply_3, voxel_size)

        # # Registro global con RANSAC 
        # result_ransac = self.execute_global_registration(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        
        # result_ransac_Wunsch = self.execute_global_registration_Wunsch(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        # print(result_ransac)
        # self.print_matrix(result_ransac.transformation,0)
        # self.draw_registration_result(source_down, target_down, result_ransac.transformation)
        # self.draw_registration_result(scaled_source, ply_3, result_ransac.transformation)
        # self.print_matrix(result_ransac.transformation,1)
        # self.draw_registration_result(source_down, target_down, result_ransac_Wunsch.transformation)
        # self.draw_registration_result(scaled_source, ply_3, result_ransac_Wunsch.transformation)


        

        
# if __name__ == '__main__':
#     option2 = PointCloudVersion()
    
#     option2.main()