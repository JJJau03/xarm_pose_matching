import open3d as o3d
mesh_path = "/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/stl/RB26_RWD_Oil_Pan_STL 3.stl"
mesh = o3d.io.read_triangle_mesh(mesh_path)
number_of_points = 20000 # Ajust this number as you find it convenient. 70000
pcd = mesh.sample_points_poisson_disk(number_of_points)
o3d.io.write_point_cloud("/home/brad/dev_ws/src/xarm_pose_matching/xarm_pose_matching/images/data/corrected.ply", pcd)
o3d.visualization.draw_geometries([pcd])