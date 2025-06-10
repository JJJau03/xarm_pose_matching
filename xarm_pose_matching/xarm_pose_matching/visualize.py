import open3d as o3d

# Cargar los archivos PLY
ply1 = o3d.io.read_point_cloud("/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/scene_v1.ply")
#ply2 = o3d.io.read_point_cloud("../images/data/rgb_image_V2.ply")

# Opcional: mover uno de los archivos para que no se sobrepongan
#ply2.translate((1.0, 0, 0))  # Mueve ply2 un metro a la derecha

# Visualizar ambos archivos
o3d.visualization.draw_geometries([ply1],
                                  window_name='Dos archivos PLY',
                                  width=800,
                                  height=600,
                                  point_show_normal=False)
    