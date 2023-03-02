from PIL import Image
import os
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def test():
    cwd=os.getcwd()
    example_dir=os.path.join(cwd,'examples')
    I0_arr=np.subtract(np.array(Image.open(os.path.join(example_dir,f'img_0.tif'))),np.array(Image.open(os.path.join(example_dir,f'img_2.tif')))).astype(int)
    I_arr=np.subtract(np.array(Image.open(os.path.join(example_dir,f'img_1.tif'))),np.array(Image.open(os.path.join(example_dir,f'img_3.tif')))).astype(int)
    plt.hist(I_arr.flatten(),bins=200)
    plt.yscale('log')
    plt.xlabel('Photon count')
    plt.ylabel('Pixel count')
    plt.savefig(os.path.join(cwd,'photon_count_threshold','photon_counts.png'),dpi=300)
    plt.close()
    for i in np.linspace(100,2000,20):
    # for i in [700]:
        photon_thresh=int(i)
        I_div_arr=np.minimum(np.divide(I0_arr,I_arr),1)
        I_div_arr[np.logical_or(I0_arr<photon_thresh,I_arr<photon_thresh)]=1
        OD_arr=-np.log(I_div_arr)
        check_arr=np.zeros_like(I0_arr)
        check_arr[np.logical_or(I0_arr<photon_thresh,I_arr<photon_thresh)]=1
        alphas=np.zeros(OD_arr.shape)
        alphas[np.where(OD_arr==0)]=1
        # alphas=np.full(check_arr.shape,(0,0,0,0))
        # alphas[alphas==1]=(0,0,0,1)
        # mycmap = mcolors.LinearSegmentedColormap.from_list('mycmap', alphas)
        fig, ax = plt.subplots()
        img1=ax.imshow(OD_arr, cmap='hsv', interpolation='none')
        ax.imshow(OD_arr, alpha=alphas, cmap='gist_gray', interpolation='none')
        plt.colorbar(img1)
        # plt.savefig(os.path.join(cwd,'photon_count_threshold',f'{photon_thresh}.png'),dpi=300)
        plt.show()
        plt.close()
        print('peak_OD:',np.max(OD_arr))
        # print('peak_OD:',np.min(OD_arr))
    return

def check():
    example_dir='/Users/Ruobin/Desktop/YALE/Spring2023/PHYS382L/Laser_cooling/examples'
    OD_arr=np.array(Image.open(os.path.join(example_dir,f'opticalDensity.tif')))
    print(np.max(OD_arr))
    plt.imshow(OD_arr, cmap='hot', interpolation='nearest')
    plt.show()
    return

def example():

    def normal_pdf(x, mean, var):
        return np.exp(-(x - mean)**2 / (2*var))


    # Generate the space in which the blobs will live
    xmin, xmax, ymin, ymax = (0, 100, 0, 100)
    n_bins = 100
    xx = np.linspace(xmin, xmax, n_bins)
    yy = np.linspace(ymin, ymax, n_bins)

    # Generate the blobs. The range of the values is roughly -.0002 to .0002
    means_high = [20, 50]
    means_low = [50, 60]
    var = [150, 200]

    gauss_x_high = normal_pdf(xx, means_high[0], var[0])
    gauss_y_high = normal_pdf(yy, means_high[1], var[0])

    gauss_x_low = normal_pdf(xx, means_low[0], var[1])
    gauss_y_low = normal_pdf(yy, means_low[1], var[1])

    weights = (np.outer(gauss_y_high, gauss_x_high)
            - np.outer(gauss_y_low, gauss_x_low))

    # We'll also create a grey background into which the pixels will fade
    greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)

    # First we'll plot these blobs using ``imshow`` without transparency.
    vmax = np.abs(weights).max()
    imshow_kwargs = {
        'vmax': vmax,
        'vmin': -vmax,
        'cmap': 'RdYlBu',
        'extent': (xmin, xmax, ymin, ymax),
    }

    alphas = np.ones(weights.shape)
    alphas[:, 30:] = np.linspace(1, 0, 70)
    print(alphas)
    fig, ax = plt.subplots()
    ax.imshow(greys)
    ax.imshow(weights, alpha=alphas, **imshow_kwargs)
    ax.set_axis_off()
    plt.show()

def rms_radius():
    print('=======START RMS RADIUS===============')
    #when calculating RMS distance, threshold for OD set to 0.25. Should probably test if varying OD threshold matters.
    photon_thresh=1000
    cwd=os.getcwd()
    tif_dir=os.path.join(cwd,'Feb_28_detuning_changed')
    volt_dict={}
    for i in range(106,132):
        volt_dict[i]=-3+0.2*(i-106)
    for key in volt_dict.keys():
        print(f'filename: {key}')
        I0_arr=np.subtract(np.array(Image.open(os.path.join(tif_dir,f'image_{key}_0.tif'))),np.array(Image.open(os.path.join(tif_dir,f'image_{key}_2.tif')))).astype(int)
        I_arr=np.subtract(np.array(Image.open(os.path.join(tif_dir,f'image_{key}_1.tif'))),np.array(Image.open(os.path.join(tif_dir,f'image_{key}_3.tif')))).astype(int)
        plt.hist(I_arr.flatten(),bins=200)
        plt.yscale('log')
        plt.xlabel('Photon count')
        plt.ylabel('Pixel count')
        # plt.savefig(os.path.join(cwd,'photon_count_threshold','photon_counts.png'),dpi=300)
        plt.show()
        plt.close()
        
        I_div_arr=np.minimum(np.divide(I0_arr,I_arr),1)
        I_div_arr[np.logical_or(I0_arr<photon_thresh,I_arr<photon_thresh)]=1
        OD_arr=-np.log(I_div_arr)
        fig, ax = plt.subplots()
        img1=ax.imshow(OD_arr, cmap='hsv', interpolation='none')
        plt.colorbar(img1)
        plt.show()
        plt.close()

        max_ind=np.unravel_index(np.argmax(OD_arr,axis=None), OD_arr.shape)
        print(f'max OD: {OD_arr[max_ind]}')
        if OD_arr[max_ind]<1:
            continue
        # print(max_ind)
        numrows,numcols=OD_arr.shape
        OD_thresholds=np.linspace(0,2,21)
        rms_dist=[]
        for OD_thresh in OD_thresholds:
            print(OD_thresh)
            weighted_dist_squared=0
            particle_total=0
            for i in range(numrows):
                for j in range(numcols):
                    particle_num=OD_arr[i,j]
                    if particle_num>=OD_thresh:
                        particle_total+=particle_num
                        weighted_dist_squared+=particle_num*((i-max_ind[0])**2+(j-max_ind[1])**2)
            print(particle_total)
            if particle_total==0:
                rms_dist.append(np.nan)
            else:
                rms_dist.append(weighted_dist_squared**0.5/particle_total)
        plt.figure()
        plt.plot(OD_thresholds,rms_dist)
        plt.show()
        plt.close()
    return

rms_radius()
# example()
# test()
# check()