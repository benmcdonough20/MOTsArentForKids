import os
#image processing
from skimage.io import imread 
from skimage.measure import (
    label, 
    regionprops
)
from scipy.ndimage import median_filter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Rectangle
from scipy.optimize import curve_fit
from scipy.stats.contingency import margins #compute marginals
from alive_progress import alive_bar #progress bar
from scipy.stats import moment

#ignore divide warnings
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

class Experiment:
    """
    Process all of the data runs in one experiment. Aggregate data.
    """

    def __init__(
        self, 
        idx_start, #starting image index
        datapath, #(string) path to folder containing images
        numtrials, #number of repeats

        ##optional constants passed to each data run. Tweak if there are issues.
        vstart=None, #ind. variable initial value
        vend=None, #ind. variable final value
        vstep=None, #ind. variable increment       
        vlist = None,
        mask = .25,
        blob_dim = 250, #size of blob box to fit
        box = 3, #size of box for image median filter
        mask_box = 50, #size of box for image mask
        circle = [(530, 675), 250], #center/radius of viewport circle
        avg_area = (200, 75, 300, 125), #area for capturing background noise
    ):
        self.vlist=vlist
        self.numtrials = numtrials
        self.idx_start = idx_start
        self.datapath = datapath

        self.args = {
            "mask" : mask,
            "blob_dim" : blob_dim,
            "box" : box,
            "mask_box" : mask_box,
            "circle" : circle,
            "avg_area" : avg_area
        }
        if self.vlist==None:
            try:
                self.vlist = [ #list of ind. variable values between initial and final
                    vstart+i*vstep 
                    for i in range(
                        round(
                            (vend-vstart)/vstep+1
                            )
                        )
                ]
            except:
                print('One of vstart, vstop, vstep is None, or vlist is None.')

        self.data = [] #data runs are stored here
        self.run() #run experiment

    def run(self):

        trials = self.vlist*self.numtrials

        with alive_bar(len(trials), force_tty=True) as bar: #progress bar
            for i,val in enumerate(trials):
                try:
                    new_dat = DataRun(
                        os.path.join(self.datapath,f"image_{self.idx_start+i}"), 
                        val, 
                        **self.args
                    )
                    self.data.append(new_dat)
                except Exception as ex:
                    print(ex)
                bar() #update progress bar
    
    def structure_data(self, func = None, remove_outliers = False):
        """Generates a dictionary where keys are independent variable values and the values
        are data collected from each run.

        Args:
            func (method): Method that takes in a DataRun and outputs desired parameter,
            such as lambda run : run.popt_x[2]
        """

        structured_data = {time:[] for time in self.vlist}

        for datum in self.data:
            if func:
                structured_data[datum.value].append(func(datum))
            else:
                structured_data[datum.value].append(datum)

        return structured_data

class DataRun:
    """Collects data from a set of four images, isolates MOT and fits marginals to Gaussian
    """

    DISTANCE_SCALE = 6.45e-6*3

    def __init__(
        self, 
        im_path, #path to image without trailing number, e.g ./data_dir/image_123
        value, #value of independent variable
        mask = .2, #threshold for mask filter
        blob_dim = 250, #size of blob box to fit
        box = 3, #size of box for image median filter
        mask_box = 50, #size of box for image mask
        circle = [(530, 690), 250], #center/radius of viewport circle
        avg_area = (200, 75, 300, 125) #area for capturing background noise
    ):
        self.value = value
        self.im_path = im_path
        self.mask = mask
        self.blob_dim = blob_dim
        self.box = box
        self.mask_box = mask_box
        self.circle = circle
        self.avg_area = avg_area

        self.xaxis, self.yaxis = [],[]
        self.od_arr = self.load() #load array of od values
        self.find_blob() #isolate blob
        self.fit() #fit marginals

    def incircle(self, center, radius, pt):
            return (pt[0]-center[0])**2 + (pt[1]-center[1])**2 < radius**2
    
    def circle_mask(self, arr):
        center,rad =self.circle
        Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
        dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
        return dist_from_center < rad
    
    def images(self):
        images =  [imread(self.im_path + f"_{i}.tif") for i in range(4)]
        im0, im1, im0_background, im1_background = images

        I0_arr = np.subtract(np.array(im0), np.array(im0_background)).astype(int)
        I_arr = np.subtract(np.array(im1), np.array(im1_background)).astype(int)

        return I_arr, I0_arr

    def load(self):
        I_arr, I0_arr = self.images()
        od_arr = np.log(np.divide(I_arr, I0_arr))
        mask=self.circle_mask(od_arr)
        #first pass, just clip anything not within the aperture
        od_arr[~mask] = 0
        od_arr=np.maximum(od_arr,0)

        #cut off the sides
        center, rad = self.circle
        od_arr = od_arr[
            center[0]-rad : center[0]+rad, 
            center[1]-rad : center[1]+rad
        ]

        avg_rect = od_arr[
            self.avg_area[1]:self.avg_area[3],
            self.avg_area[0]:self.avg_area[2]
        ]
        od_arr = od_arr - np.mean(avg_rect)
        
        od_arr = median_filter(od_arr, self.box)

        return od_arr

    def find_blob(self):
        value_mask = self.od_arr > self.mask
        self.mask_filtered = median_filter(value_mask, self.mask_box)

        blobs = label(self.mask_filtered)
        props = regionprops(blobs) #generate a properties dictionary
        if not len(props) == 1:
            raise Exception()
        self.cy, self.cx = props[0].centroid
        
        self.blob = self.od_arr[
            round(self.cy-self.blob_dim/2):round(self.cy+self.blob_dim/2), 
            round(self.cx-self.blob_dim/2):round(self.cx+self.blob_dim/2)
        ]

        self.yaxis = np.arange(len(self.blob))*self.DISTANCE_SCALE
        self.xaxis = np.arange(len(self.blob[0]))*self.DISTANCE_SCALE

    def gaussian_fit(self, x, A, mu, sigma):
        return A*np.exp(-(x-mu)**2/(2*sigma**2))
    
    def fit(self):
        #compute marginals and fit to a gaussian
        y, x = margins(self.blob)
        x = x[0]
        y = y.T[0]

        x[x == np.inf] = 0
        y[y == np.inf] = 0

        self.popt_x, self.pcov_x = curve_fit(
            self.gaussian_fit, 
            self.xaxis,
            x, 
            p0 = [350, .00288, .00117]
        )

        self.popt_y, self.pcov_y = curve_fit(
            self.gaussian_fit, 
            self.yaxis,
            y, 
            [350, .00288, .00117]
        )

        self.x = x
        self.y = y

    def atom_number(self):
        x = np.arange(-10*self.popt_x[2], 10*self.popt_x[2], self.DISTANCE_SCALE/20)

        abs_CS = (766.5e-9)**2/(2*np.pi)
        #of magnitude smaller
        return np.trapz(
            self.gaussian_fit(x, *self.popt_x[:3], 0),
            x
        )*self.DISTANCE_SCALE/abs_CS

    def atom_number_px_sum(self):
        abs_CS=(766.5e-9)**2/(2*np.pi)
        return np.sum(self.blob)*self.DISTANCE_SCALE**2/abs_CS 
        
    def plot_blob(self):
        fig, ax = plt.subplots()

        blob_rect = (
            self.cx-self.blob_dim/2,
            self.cy-self.blob_dim/2, 
            self.cx+self.blob_dim/2, 
            self.cy+self.blob_dim/2
        )

        rect1 = Rectangle(
            (blob_rect[0],blob_rect[1]),
            (blob_rect[2]-blob_rect[0]),
            (blob_rect[3]-blob_rect[1]), 
            fill=False,
            color = "r"
        )
        rect2 = Rectangle(
            (self.avg_area[0],self.avg_area[1]),
            (self.avg_area[2]-self.avg_area[0]),
            (self.avg_area[3]-self.avg_area[1]), 
            fill=False,
            color = "g"
        )

        ax.add_artist(rect1)
        ax.add_artist(rect2)


        im = ax.imshow(self.od_arr, extent = [0, len(self.od_arr[0])*self.DISTANCE_SCALE*1000, 0, len(self.od_arr)*self.DISTANCE_SCALE*1000])
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im)

    def plot_fit(self):

        fig = plt.figure(
            figsize=(6,6)
        )
        gs = fig.add_gridspec(
            2, 2,  
            width_ratios=(4, 1), height_ratios=(1, 4),
            left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.3, hspace=0.3
        )
        ax = fig.add_subplot(gs[1,0])

        ax.imshow(self.blob, extent=[0, max(self.xaxis), 0, max(self.yaxis)])

        ax_x = fig.add_subplot(gs[0,0], sharex=ax)
        ax_y = fig.add_subplot(gs[1,1], sharey=ax)


        ax_x.plot(self.xaxis*1000, self.x, color = "red")
        ax_x.plot(
            self.xaxis*1000,
            self.gaussian_fit(self.xaxis,*self.popt_x),
            color = "blue"
        )

        #flip the axes for y
        ax_y.plot(self.y, self.yaxis*1000, color = "red") 
        ax_y.plot(
            self.gaussian_fit(self.yaxis, *self.popt_y),
            self.yaxis*1000,
            color = "blue"
        )

        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")

        ax.errorbar(
            self.popt_x[1], 
            self.popt_y[1], 
            xerr = np.abs(self.popt_x[2]), 
            yerr = np.abs(self.popt_y[2]), 
            color = 'r', 
            marker = "x", 
            capsize = 10
        )
