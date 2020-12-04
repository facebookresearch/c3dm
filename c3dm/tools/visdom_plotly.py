# Copyright (c) Facebook, Inc. and its affiliates.

from visdom import Visdom
from tools.vis_utils import get_visdom_connection, denorm_image_trivial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from PIL import Image



fig = make_subplots(
            rows = 1, cols = 1, 
            specs=[[{"type": "scene"}]],
            subplot_titles=(title),
            column_widths=[0.5],
        )
fig.add_trace(
    go.Scatter3d(
        x=-pt_cloud_np[:, 0],
        y=-pt_cloud_np[:, 1],
        z=-pt_cloud_np[:, 2],
        mode='markers',
        name=k,
        visible=True,
        marker=dict(
            size=8,
            color=color,
            opacity=1.,
        )), row = 0, col = 0)





class VisdomPlotly():
    def __init__(self, viz, visdom_env_imgs=None, win=None):
        
        self.camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=2.0, z=0.0)
        )

        self.scene = dict(
            xaxis = dict(nticks=10, range=[-100,100],),
            yaxis = dict(nticks=10, range=[-100,100],),
            zaxis = dict(nticks=10, range=[-100,100],),
            camera = self.camera)

    def extend_to_skeleton(self, pt_cloud, skeleton, line_resolution = 25):
        ptcloud_now = pt_cloud
        for stick in skeleton:
            alpha = np.linspace(0,1,line_resolution)[:, None]
            linepoints = pt_cloud[stick[0],:][None,:] * alpha + \
                        pt_cloud[stick[1],:][None,:] * ( 1. - alpha )
            ptcloud_now = np.concatenate((ptcloud_now,linepoints),axis=0)

        return ptcloud_now

    def make_fig(self, rows, cols, epoch, it, idx_image, acc_detail, percent_agree):
        # fig_dict['subplot_title']
        title="e%d_it%d_im%d"%(epoch, it, idx_image)

        self.fig = make_subplots(
            rows = rows, cols = cols, 
            specs=[[{"type": "xy"},{"type": "xy"},{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "Input: {0}".format(title),
                "Projection",
                acc_detail,
                'Mode Freqs',
                'Mode Freqs (Flow): {0}'.format(percent_agree)
            ),
            column_widths=[0.5] * cols,
        )


# vis_plot = VisdomPlotly(visdom_env_imgs, stats.visdom_server, stats.visdom_port)
# vis_plot.make_fig(1, 5, stats.epoch, stats.it[trainmode], idx_image, "sqerr [M{0}]: {1:.2f}".format(min_mode, rmse_h36m), flow_agree)

# vis_plot.add_image(img_with_gt)
# vis_plot.add_2d_points(keypoints_2d.reshape(-1, 2), 1, 1, 'Input (Joints)', 'green')

# vis_plot.add_2d_points(keypoints_2d.reshape(-1, 2), 1, 2, 'Input (Joints)', 'green')

# vis_plot.add_3d_points(gt_sample.reshape(-1, 3) * 0.1, 1, 3, 'GT', 'green', visible='legendonly')
# vis_plot.add_3d_points(in_verts[idx_image].reshape(-1, 3) * 0.1, 1, 3, 'GT', 'green', s=1, opacity=0.5)
 
    def add_image(self, img):
        bg_image = Image.fromarray(img)

        self.fig.update_layout(
            images = [
                go.layout.Image(
                    source=bg_image,
                    xref="x1",
                    yref="y1",
                    x=0,
                    y=bg_image.size[1],
                    sizex=bg_image.size[0],
                    sizey=bg_image.size[1],
                    sizing="stretch",
                    opacity=0.75,
                    layer="below"),

                go.layout.Image(
                    source=bg_image,
                    xref="x2",
                    yref="y2",
                    x=0,
                    y=bg_image.size[1],
                    sizex=bg_image.size[0],
                    sizey=bg_image.size[1],
                    sizing="stretch",
                    opacity=0.75,
                    layer="below")
            ]
        )

    def add_3d_points(self, pt_cloud_np, row, col, name, color, opacity=1.0, s=8, visible=True):
        
        self.fig.add_trace(
            go.Scatter3d(
                x=-1 * pt_cloud_np[:, 0],
                y=-1 * pt_cloud_np[:, 2],
                z=-1 * pt_cloud_np[:, 1],
                mode='markers',
                name=name,
                visible=visible,
                marker=dict(
                    size=s,
                    color=color,
                    opacity=opacity,
                )), row = row, col = col)

        self.fig.update_scenes(patch = self.scene, row = row, col = col)
        self.add_hack_points(row, col)


    # def add_mesh(self, verts, triangles, row, col, name, color):
    #     self.fig.add_trace(
    #         go.Mesh3d(
    #             x=verts[:, 0],
    #             y=verts[:, 1],
    #             z=verts[:, 2],
    #             colorbar_title='z',
    #             colorscale=[[0, 'gold'], 
    #                         [0.5, 'mediumturquoise'], 
    #                         [1, 'magenta']],
    #             # Intensity of each vertex, which will be interpolated and color-coded
    #             intensity=[0, 0.33, 0.66, 1],
    #             # i, j and k give the vertices of triangles
    #             i=triangles[:, 0],
    #             j=triangles[:, 1],
    #             k=triangles[:, 2],
    #             name=name,
    #             showscale=True
    #         )
    #     )
    #     self.fig.update_scenes(patch = self.scene, row = row, col = col)

    def add_2d_points(self, points, row, col, name, color, scale=6, opacity=1.0, im_size = 224, extend=False, visible=True):
        points_npy = points

        if extend:
            points_npy = self.extend_to_skeleton(points_npy, SKELETON_2D)
                
        self.fig.add_trace(
            go.Scatter(
            x=points_npy[:, 0],
            y=im_size-points_npy[:, 1],
            mode='markers',
            name=name,
            visible=visible,
            marker=dict(
                size=scale,
                color=color,                # set color to an array/list of desired values
                opacity=opacity,
            )), row = row, col = col)

        self.fig.update_xaxes(range=[0, im_size], row=row, col=col)
        self.fig.update_yaxes(range=[0, im_size], row=row, col=col)

    def show(self):
        raw_size = 400
        self.fig.update_layout(height = raw_size, width = raw_size * 5)
        self.viz.plotlyplot(self.fig, env=self.visdom_env_imgs)

    def add_hack_points(self, row, col):
        hack_points = np.array([
            [-1000.0, -1000.0, -1000.0],
            [-1000.0, -1000.0, 1000.0],
            [-1000.0, 1000.0, -1000.0],
            [-1000.0, 1000.0, 1000.0],
            [1000.0, -1000.0, -1000.0],
            [1000.0, -1000.0, 1000.0],
            [1000.0, 1000.0, -1000.0],
            [1000.0, 1000.0, 1000.0]])

        self.fig.add_trace(
            go.Scatter3d(
                x=-1 * hack_points[:, 0],
                y=-1 * hack_points[:, 2],
                z=-1 * hack_points[:, 1],
                mode='markers',
                name='_fake_pts',
                visible=False,
                marker=dict(
                    size=1,
                    opacity = 0,
                    color=(0.0, 0.0, 0.0),
                )), row = row, col = col)

    def add_bar(self, stats, num_modes, row, col, name):
        freqs = np.bincount(stats, minlength=num_modes)
        fig = self.fig.add_trace(
            go.Bar(
                x=list(range(num_modes)), 
                y=freqs, 
                name=name), row = row, col = col)