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
        vstart, #ind. variable initial value
        vend, #ind. variable final value
        vstep, #ind. variable increment
        idx_start, #starting image index
        datapath, #(string) path to folder containing images
        numtrials, #number of repeats

        ##optional constants passed to each data run. Tweak if there are issues.        

        mask = .1, #threshold for mask filter
        blob_dim = 250, #size of blob box to fit
        box = 3, #size of box for image median filter
        mask_box = 50, #size of box for image mask
        circle = [(530, 690), 250], #center/radius of viewport circle
        avg_area = (200, 75, 300, 125) #area for capturing background noise
    ):

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

        self.vlist = [ #list of ind. variable values between initial and final
            vstart+i*vstep 
            for i in range(
                round(
                    (vend-vstart)/vstep+1
                    )
                )
        ]

        self.data = [] #data runs are stored here
        self.run() #run experiment

    def run(self):

        trials = self.vlist*self.numtrials

        with alive_bar(len(trials), force_tty=True) as bar: #progress bar
            for i,val in enumerate(trials):
                self.data.append(
                    DataRun(
                        os.path.join(self.datapath,f"image_{self.idx_start+i}"), 
                        val, 
                        **self.args
                        )
                )
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

    DISTANCE_SCALE = 6.45e-6/3

    def __init__(
        self, 
        im_path, #path to image without trailing number, e.g ./data_dir/image_123
        value, #value of independent variable
        mask = 0.1, #threshold for mask filter
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

    def images(self):
        images =  [imread(self.im_path + f"_{i}.tif") for i in range(4)]
        im0, im1, im0_background, im1_background = images

        I0_arr = np.subtract(np.array(im0), np.array(im0_background)).astype(int)
        I_arr = np.subtract(np.array(im1), np.array(im1_background)).astype(int)

        return I_arr, I0_arr

    def load(self):
        I_arr, I0_arr = self.images()
        od_arr = np.log(np.divide(I_arr, I0_arr))

        #first pass, just clip anything not within the aperture
        for i,row in enumerate(od_arr):
            for j, pixel in enumerate(row):
                if pixel < 0 or not self.incircle(*self.circle,(i,j)):
                    od_arr[i][j] = 0

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


    def gaussian_fit(self, x, A, mu, sigma, B):
        return A*np.exp(-(x-mu)**2/(2*sigma**2))+B
    
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
            p0 =[350, 3.2e-4, 1.3e-4, 0]
        )

        self.popt_y, self.pcov_y = curve_fit(
            self.gaussian_fit, 
            self.yaxis,
            y, 
            [350,3.2e-4, 1.3e-4, 0]
        )

        self.x = x
        self.y = y

    def atom_number(self):
        x = np.arange(-3*self.popt_x[2], 3*self.popt_x[2], self.DISTANCE_SCALE/2)
        abs_CS=(766.5e-9)**2/(2*np.pi)

        return 1/abs_CS*np.trapz(
            self.gaussian_fit(x, *self.popt_x[:3], 0),
            x
        ) *\
        np.trapz(
            self.gaussian_fit(x, *self.popt_y[:3], 0),
            x
        )
    def atom_number_px_sum_noise_cancelled(self):
        abs_CS=(766.5e-9)**2/(2*np.pi)
        return np.sum(self.blob-0.5*(self.popt_x[3]+self.popt_y[3]))*self.DISTANCE_SCALE**2/abs_CS
    
    def atom_number_px_sum(self):
        abs_CS=3*(766.5e-9)**2/(2*np.pi)
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
        im = ax.imshow(self.od_arr)
        
        fig.colorbar(im)
        fig.set_dpi(200)

    def plot_fit(self):

        fig = plt.figure(
            figsize=(6,6),
            dpi = 200
        )
        gs = fig.add_gridspec(
            2, 2,  
            width_ratios=(4, 1), height_ratios=(1, 4),
            left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.05, hspace=0.05
        )
        ax = fig.add_subplot(gs[1,0])

        ax.imshow(self.blob, extent = (0, max(self.xaxis), 0, max(self.yaxis)))

        ax_x = fig.add_subplot(gs[0,0], sharex=ax)
        ax_y = fig.add_subplot(gs[1,1], sharey=ax)

        ax_x.plot(self.xaxis, self.x, color="r")
        ax_x.plot(
            self.xaxis,
            self.gaussian_fit(self.xaxis, *self.popt_x),
            color = "b"
        )
        ax_x.set_ylabel("OD (dim.)")
        plt.setp(ax_x.get_xticklabels(), visible = False)

        #flip the axes for y
        ax_y.plot(self.y, self.yaxis, color = "r") 
        ax_y.plot(
            self.gaussian_fit(self.yaxis, *self.popt_y),
            self.yaxis,
            color = "b"
        )
        ax_y.set_xlabel("OD (dim.)")
        plt.setp(ax_y.get_yticklabels(), visible = False)

        ax.errorbar(
            self.popt_x[1], 
            self.popt_y[1], 
            xerr = np.abs(self.popt_x[2]), 
            yerr = np.abs(self.popt_y[2]), 
            color = "r", 
            marker = "x", 
            capsize = 10
        )

        def tick_format(val, num):
            return "{:.2f}".format(val * 1e3)

        ax.xaxis.set_major_formatter(tick_format)
        ax.yaxis.set_major_formatter(tick_format)

        ax.set_xlabel(r"$x \ (mm)$")
        ax.set_ylabel(r"$y \ (mm)$")
