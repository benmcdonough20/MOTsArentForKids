from PIL import Image
import os
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
def test():
    example_dir='/Users/Ruobin/Desktop/YALE/Spring2023/PHYS382L/Laser_cooling/examples'
    I0_arr=np.subtract(np.array(Image.open(os.path.join(example_dir,f'img_0.tif'))),np.array(Image.open(os.path.join(example_dir,f'img_2.tif')))).astype(int)
    I_arr=np.subtract(np.array(Image.open(os.path.join(example_dir,f'img_1.tif'))),np.array(Image.open(os.path.join(example_dir,f'img_3.tif')))).astype(int)
    # plt.hist(I_arr.flatten(),bins=200)
    # plt.show()
    # plt.close()
    I_div_arr=np.minimum(np.divide(I0_arr,I_arr),1) 
    I_div_arr[np.logical_or(I0_arr<1000,I_arr<1000)]=1
    OD_arr=-np.log(I_div_arr)
    plt.imshow(OD_arr, cmap='hot', interpolation='nearest')
    # plt.legend()
    plt.show()
    print('peak_OD:',np.max(OD_arr))
    # print('peak_OD:',np.min(OD_arr))
    return
test()

def check():
    example_dir='/Users/Ruobin/Desktop/YALE/Spring2023/PHYS382L/Laser_cooling/examples'
    OD_arr=np.array(Image.open(os.path.join(example_dir,f'opticalDensity.tif')))
    print(np.max(OD_arr))
    plt.imshow(OD_arr, cmap='hot', interpolation='nearest')
    plt.show()
    return
# check()


