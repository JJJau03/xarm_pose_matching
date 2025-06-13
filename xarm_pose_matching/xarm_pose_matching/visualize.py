import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy

# Cargar los archivos PLY
ply1 = o3d.io.read_point_cloud("/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply")
#ply2 = o3d.io.read_point_cloud("../images/data/rgb_image_V2.ply")

# Opcional: mover uno de los archivos para que no se sobrepongan

#rotación en x
ply2 = copy.deepcopy(ply1)
ply2.translate((1.0, 0, 0))  # Mueve ply2 un metro a la derecha
#ply1.rotate(ply1.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)), center=(0, 0, 0))
#ply2.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
#rotación en y 
#ply1.rotate(ply1.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)), center=(0, 0, 0))
# Visualizar ambos archivos
o3d.visualization.draw_geometries([ply1],
                                  window_name='Dos archivos PLY',
                                  width=800,
                                  height=600,
                                  point_show_normal=False)
    