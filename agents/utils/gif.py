import torch
import numpy as np
from PIL import Image
import os

def generate_gif(
    env, 
    filepath, 
    pi, 
    device, 
    max_episode_steps, 
    resize_to=None, 
    duration=32
):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment
    filepath : str
    pi : nn.Module
    max_episode_steps : int
    resize_to : tuple of ints, optional
    duration : float, optional
    """
    
    # collect frames
    frames = []
    s = env.reset()
    for t in range(max_episode_steps):
        a = pi.get_action(torch.Tensor(s).to(device))
        s_next, r, done, info = env.step(a)
        # store frame
        frame = env.render(mode='rgb_array')
        frame = Image.fromarray(frame)
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render(mode='rgb_array')
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    frames[0].save(
        fp=filepath, 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True,
        duration=duration, 
        loop=0
    )
