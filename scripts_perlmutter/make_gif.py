from PIL import Image
import glob
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--folder', default='../plots_closure_pct', help='Folder containing figures')
parser.add_argument('--plot', default='gen_jet_tau15', help='Name of the distribution to plot')
parser.add_argument('--niter', type=int, default=5, help='Number of iterations to run over')
flags = parser.parse_args()


# Create the frames
frames = []
base_folder = flags.folder
to_gif = flags.plot



for i in range(flags.niter):
    new_frame = Image.open(os.path.join(base_folder,"{}_{}.png".format(to_gif,i)))
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save(os.path.join(base_folder,'{}.gif'.format(to_gif)), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=500, loop=0)
