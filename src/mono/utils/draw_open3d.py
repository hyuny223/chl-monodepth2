import open3d
import numpy as np
from . import process_open3d

class Open3D:
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        # self.vis = open3d.visualization.VisualizerWithEditing()
        self.vis.create_window()
        self.vis.add_geometry(None)
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([0, 0, 0])

        control = open3d.visualization.ViewControl()
        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(step=10)

    '''
    uncompleted
    '''
    def drawLiDAR(self, ros_cloud):

        c = process_open3d.convertCloudFromRosToOpen3d(ros_cloud)
        open3d.visualization.draw_geometries([c],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])

        # self.vis.add_geometry(c)
        # self.vis.update_geometry(c) #informs the `vis` that the related geometries are updated
        # self.vis.poll_events()  #visualizer renders a new frame by calling `poll_events` and `update_renderer`.
        # self.vis.update_renderer()



    def drawImage(self, open3d_cloud):
        # open3d.visualization.draw_geometries([open3d_cloud])

        open3d.visualization.draw_geometries([open3d_cloud])

        # self.vis.add_geometry(open3d_cloud)
        # self.vis.update_geometry(open3d_cloud) #informs the `vis` that the related geometries are updated
        # self.vis.poll_events()  #visualizer renders a new frame by calling `poll_events` and `update_renderer`.
        # self.vis.update_renderer()


