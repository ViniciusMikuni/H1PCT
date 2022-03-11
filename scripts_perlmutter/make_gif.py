from PIL import Image
import glob
import os
import argparse


parser = argparse.ArgumentParser()

#parser.add_argument('--folder', default='../plots_Rapgap_nominal_closure_hybrid', help='Folder containing figures')
parser.add_argument('--folder', default='../plots_Rapgap_nominal_sys_hybrid', help='Folder containing figures')
#parser.add_argument('--plot', default='gen_jet_tau15', help='Name of the distribution to plot')
parser.add_argument('--niter', type=int, default=4, help='Number of iterations to run over')
flags = parser.parse_args()


# Create the frames

base_folder = flags.folder
#to_gif = flags.plot

plot_list = {
    'gen_jet_ncharged_6':r'$\mathrm{N_{c}}$', 
    'gen_jet_charge_6':r'$\mathrm{Q_1}$', 
    'gen_jet_ptD_6':r'$p_\mathrm{T}\mathrm{D}$',
    'gen_jet_tau10_6':r'$\mathrm{log}(\tau_{1})$', 
    'gen_jet_tau15_6':r'$\mathrm{log}(\tau_{0.5})$',
    'gen_jet_tau20_6':r'$\mathrm{log}(\tau_{0})$'}


for to_gif in plot_list:
    frames = []
    for i in range(1,flags.niter+1):
        new_frame = Image.open(os.path.join(base_folder,"{}_{}.png".format(to_gif,i)))
        frames.append(new_frame)
 
    # Save into a GIF file that loops forever
    frames[0].save(os.path.join(base_folder,'{}.gif'.format(to_gif)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)
