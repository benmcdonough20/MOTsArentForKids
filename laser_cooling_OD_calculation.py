from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

wd = Path(os.getcwd())

def test():
    example_dir = wd / "test_data"
    I0_arr=np.subtract(np.array(Image.open(example_dir / 'img_0.tif')),
                       np.array(Image.open( example_dir / 'img_2.tif'))).astype(int)
    I_arr=np.subtract(np.array(Image.open(example_dir / 'img_1.tif')),
                      np.array(Image.open(example_dir / 'img_3.tif'))).astype(int)
    plt.hist(I_arr.flatten(),bins=200)
    plt.yscale('log')
    plt.xlabel('Photon count')
    plt.ylabel('Pixel count')
    plt.savefig(wd / 'photon_count_threshold/photon_counts.png',dpi=300)
    plt.close()

    for i in np.linspace(100,2000,20):

        photon_thresh=int(i)
        I_div_arr=np.minimum(np.divide(I0_arr,I_arr),1)
        I_div_arr[np.logical_or(I0_arr<photon_thresh,I_arr<photon_thresh)]=1
        OD_arr=-np.log(I_div_arr)
        check_arr=np.zeros_like(I0_arr)
        check_arr[np.logical_or(I0_arr<photon_thresh,I_arr<photon_thresh)]=1
        alphas=np.zeros(OD_arr.shape)
        alphas[np.where(OD_arr==0)]=1
        fig, ax = plt.subplots()
        img1=ax.imshow(OD_arr, cmap='hsv', interpolation='none')
        ax.imshow(OD_arr, alpha=alphas, cmap='gist_gray', interpolation='none')
        plt.colorbar(img1)

        plt.show()
        plt.close()
        print('peak_OD:',np.max(OD_arr))

    return

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

#rms_radius()
#example()
test()
# check()