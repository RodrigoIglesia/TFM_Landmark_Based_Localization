import numpy as np
import open3d as o3d

def custom_draw_geometry_with_custom(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_front([ 0.89452025962235249, -0.18810349064390619, 0.40552506942572236 ])
    ctr.set_lookat([ -4803.5406166994117, 2578.7673925914692, 1502.733219030637 ])
    ctr.set_up([ -0.39927129067518657, 0.071776107780855442, 0.91401894225141844 ])
    ctr.set_zoom(0.16)

    vis.run()
    # vis.destroy_window()

if __name__ == "__main__":
    depth = np.squeeze(np.load('/home/rodrigo/catkin_ws/src/TFM_Landmark_Based_Localization/src/hd_map_gen/test.npy'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth)
    custom_draw_geometry_with_custom(pcd)
