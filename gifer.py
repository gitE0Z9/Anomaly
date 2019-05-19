import os
import imageio
from glob import glob

def create_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        if image_name[0]=='0':
            frames.append(imageio.imread(image_name[1:7]))
        else:
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
 
    return
 
def running(path):
    image_list = sorted([i if len(i)==6 else '0'+i for i in glob('*.png')])
    gif_name = '../10step-{}.gif'.format(path)
    create_gif(image_list, gif_name)
 
def main(path):
    os.chdir('gif/{}/{}'.format(modelname,hostname))
    for i in os.listdir():
        os.chdir(i)
        try:
            running(i)
        except:
            pass
        os.chdir('..')

