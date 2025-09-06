import glob
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from bosRegressor.utils.misc_utils import colors
from bosRegressor.utils.misc_utils import copy2cpu as c2c


def visualize_mesh(bm_output, faces, frame_id=0, display=False):
    imw, imh = 1600, 1600

    body_mesh = trimesh.Trimesh(vertices=c2c(bm_output.vertices[frame_id]),
                                faces=faces,
                                vertex_colors=np.tile(colors['grey'], (6890, 1)),
                                process=False,
                                maintain_order=True)

    if display:
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)

    return body_mesh


def plot_sequence_data(vecs, name, exp_name='test'):
    """
    Plot the first and second derivatives of the smplx pose params each in a separate plot figure, with 156 subplots.
    Save all in a folder called pose_plots
    vecs: (N, C)
    name: quantity name
    """
    out_dir = f'debug_plots/{exp_name}'
    os.makedirs(out_dir, exist_ok=True)
    # Plot separate channels in the same plot
    plt.figure()
    plt.grid()

    for i in range(vecs.shape[-1]):
        plt.plot(vecs[:, i].detach().cpu().numpy(), label=f'{name}_{i}')

    plt.xlabel('frames')
    plt.ylabel(f'{name}')
    plt.title(f'{exp_name}')
    plt.legend()
    plt.savefig(f'{out_dir}/{name}.jpg')
    print(f'Saved {out_dir}/{name}.jpg')
    plt.close()


class RealTimePlot:
    def __init__(self, y_data1, y_data2, y_data3, name=None, fps=60, exp_name='test'):
        self.name = name
        self.fps = fps

        # run a GUI event loop
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(1920 / self.fig.dpi, 400 / self.fig.dpi)
        self.ax.grid()

        x_data = np.arange(y_data1.shape[0])
        self.line1, = self.ax.plot(x_data, y_data1, label=f'{name}_x')
        self.line2, = self.ax.plot(x_data, y_data2, label=f'{name}_y')
        self.line3, = self.ax.plot(x_data, y_data3, label=f'{name}_z')
        self.x_data = x_data
        self.y_data1 = y_data1
        self.y_data2 = y_data2
        self.y_data3 = y_data3
        self.ax.legend()
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        self.ax.set_title(name)

        self.out_dir = f'debug_plots/frames/{exp_name}'
        os.makedirs(self.out_dir, exist_ok=True)

    def update_plot(self, frame):
        # update the data
        self.line1.set_xdata(self.x_data[:frame])
        self.line1.set_ydata(self.y_data1[:frame])
        self.line2.set_xdata(self.x_data[:frame])
        self.line2.set_ydata(self.y_data2[:frame])
        self.line3.set_xdata(self.x_data[:frame])
        self.line3.set_ydata(self.y_data3[:frame])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # save the figure

        self.fig.savefig(f'{self.out_dir}/{self.name}_{str(frame).zfill(4)}.jpg', dpi=self.fig.dpi)

    def make_video(self):
        image_paths = sorted(glob.glob(os.path.join(self.out_dir, f'{self.name}_*.jpg')))
        out_path = os.path.join(self.out_dir, f'{self.name}.mp4')
        # create a video with ffmpeg
        subprocess.call(
            ['ffmpeg', '-y', '-framerate', str(self.fps), '-i', os.path.join(self.out_dir, f'{self.name}_%04d.jpg'), '-c:v',
             'libx264', '-pix_fmt', 'yuv420p', out_path])
        print(f'Saved {out_path}')
        # delete all images
        for image_path in image_paths:
            os.remove(image_path)
