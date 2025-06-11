import numpy as np
import open3d as o3d

class Wunsch():
    def __init__(self):
        pass
    def refine_with_svd(source_points, target_points):
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
    
    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
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
            weights = self.compute_weights(src_pts, tgt_pts, result.transformation)
            
            # Refinar con SVD ponderado
            refined_trans = self.weighted_svd_refinement(src_pts, tgt_pts, weights)
            result.transformation = refined_trans
            
            # Actualizar correspondencias basado en la nueva transformación
            transformed_src = (refined_trans[:3,:3] @ source_points.T).T + refined_trans[:3,3]
            dists = np.linalg.norm(transformed_src[:,np.newaxis] - target_points, axis=2)
            correspondences = np.stack([np.arange(len(source_points)), np.argmin(dists, axis=1)], axis=1)
        
        return result
    def compute_weights(source_points, target_points, transformation, sigma=0.1):
        """
        Calcula pesos para cada correspondencia basado en el error
        """
        transformed_source = (transformation[:3,:3] @ source_points.T).T + transformation[:3,3]
        errors = np.linalg.norm(transformed_source - target_points, axis=1)
        weights = np.exp(-errors**2/(2*sigma**2))
        return weights

    def weighted_svd_refinement(source_points, target_points, weights):
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
        print()
