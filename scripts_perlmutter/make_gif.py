from PIL import Image
from PIL import ImageDraw,ImageFont
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
font = ImageFont.truetype("Helvetica-Bold.ttf", size=35)

plot_list = {
    'gen_jet_ncharged_6':r'$\mathrm{N_{c}}$', 
    'gen_jet_charge_6':r'$\mathrm{Q_1}$', 
    'gen_jet_ptD_6':r'$p_\mathrm{T}\mathrm{D}$',
    'gen_jet_tau10_6':r'$\mathrm{log}(\lambda_1^1)$', 
    'gen_jet_tau15_6':r'$\mathrm{log}(\lambda_{1.5}^1)$',
    'gen_jet_tau20_6':r'$\mathrm{log}(\lambda_2^1)$',
    
    # 'gen_jet_ncharged':r'$\mathrm{N_{c}}$', 
    # 'gen_jet_charge':r'$\mathrm{Q_1}$', 
    # 'gen_jet_ptD':r'$p_\mathrm{T}\mathrm{D}$',
    # 'gen_jet_tau10':r'$\mathrm{log}(\lambda_1^1)$', 
    # 'gen_jet_tau15':r'$\mathrm{log}(\lambda_{1.5}^1)$',
    # 'gen_jet_tau20':r'$\mathrm{log}(\lambda_2^1)$',
}

for to_gif in plot_list:
    frames = []
    for i in range(1,flags.niter+1):
        new_frame = Image.open(os.path.join(base_folder,"{}_{}.png".format(to_gif,i)))
        # draw = ImageDraw.Draw(new_frame)
        # draw.text((120, 75), "Iteration {}".format(i),fill="black",font=font)
        frames.append(new_frame)
 
    # Save into a GIF file that loops forever
    frames[0].save(os.path.join(base_folder,'{}.gif'.format(to_gif)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)
