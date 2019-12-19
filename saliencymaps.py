import torch
import argparse
import numpy as np
from torch.autograd import Variable
from a3c.src.common.rollouts import rollouts
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from skimage import transform

parser = argparse.ArgumentParser(description="Script to run trained agent")
parser.add_argument("--checkpoint-file", type=str, help="checkpoint file to load")
parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes to run, (default = 10)")
parser.add_argument("--render", type=int, default=1, help="render the environment (1-true, 0-false), (default = 1)")
parser.add_argument("--maps", type=int, default=0, help="renders saliency maps (1-true), (default=0)")

occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur

def run_through_model(model, history, ix, interp_func=None, mask=None, blur_memory=None, mode='actor'):
    if mask is None:
        im = history['ins'][ix]
    else:
        assert(interp_func is not None, "interp func cannot be none")
        im = interp_func(history['ins'][ix].squeeze(), mask).reshape(1, 80, 80)
    tens_state = torch.Tensor(im)
    state = Variable(tens_state.unsqueeze(0), volatile=True)
    hx = Variable(torch.Tensor(history['hx'][ix-1]).view(1, -1))
    return model((state, hx))[0] if mode == 'critic' else model((state, hx))[1]

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def saliency_on_frame(saliency, frame, fudge_factor, channel=2, sigma=0):
    pmax = saliency.max()
    print(pmax)
    S = imresize(saliency, size=[160, 160], interp="bilinear").astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= frame.min(); S = fudge_factor * pmax * S/S.max()
    I = S.astype('uint16')
    #I[:,:, channel] += S.astype('uint16')
    I = I.clip(1, 255).astype('uint8')
    return I

def score_frame(model, history, ix, r, d, interp_func, mode='actor'):
    assert mode in ['actor', 'critic'], 'mode must be either actor or critic'
    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(80/d)+1,int(80/d)+1)) # saliency scores S(t,i,j)
    for i in range(0, 80, d):
        for j in range(0, 80, d):
            mask = get_mask(center=[i,j], size=[80,80], r=r)
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode)
            #print((L-l).pow(2).sum().mul_(.5).item())
            #print(l.max(), L.max())
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).item()
    pmax = scores.max()
    scores = imresize(scores, size=[80,80], interp='bilinear').astype(np.float32)
    return pmax * scores / (scores.max() + 1e-8)



def mergedMap(image, heat_map):

    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    return normalized_heat_map
    # display
    '''
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    '''

if __name__ == "__main__":

    args = parser.parse_args()
    testEnv = rollouts(args.checkpoint_file, args.maps)

    for i in range(0, args.num_episodes):
        render = args.render == 1

        res, history = testEnv.runEpisode(render)
        print("Finished Episode : ", i+1, "Reward : ", res["cum_reward"], "Episode Length : ", res["episode_length"])
        numFrames = len(history['ins'])
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=testEnv.args.getValue('env_name'), artist='mayank', comment='RL agent saliency maps')
        writer = FFMpegWriter(fps=8, metadata=metadata)
        f = plt.figure(figsize=[6, 6*1.3], dpi=75)
        
        with writer.saving(f, "file.mp4", dpi=75):        
            for i in range(1, numFrames):
                frame = history['ins'][i].squeeze().copy()
                obs = frame.copy()
                actor_saliency = score_frame(testEnv.model, history, i, 5, 5, occlude, 'actor')
                critic_saliency = score_frame(testEnv.model, history, i, 5, 5, occlude, 'critic')
                
                heatMapActor = mergedMap(obs, actor_saliency)
                heatMapCritic = mergedMap(obs, critic_saliency)
                
                plt.imshow(heatMapActor, alpha=0.9, cmap="hot")
                plt.imshow(heatMapCritic, alpha=0.9, cmap="hot")
                plt.imshow(obs, alpha=0.7, cmap="gray")
                
                plt.title(testEnv.args.getValue('env_name').lower(), fontsize=15)
                plt.axis("off")
                plt.savefig("images/frame_%d.png"%(i+1))
                f.clear()
                print("Completed,...")


    testEnv.env.close()