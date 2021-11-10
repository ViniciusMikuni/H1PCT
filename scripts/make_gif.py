from PIL import Image
import glob
import os

# Create the frames
frames = []
base_folder = 'plots_closure_pct'
to_gif = 'gen_jet_ncharged'
niter = 14

for i in range(niter):
    new_frame = Image.open(os.path.join(base_folder,"{}_{}.png".format(to_gif,i)))
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save(os.path.join(base_folder,'ncharged.gif'), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=500, loop=0)
