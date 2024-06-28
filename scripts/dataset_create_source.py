import cv2 as cv
import time
import random
import numpy as np

#colors
'''WHITE = (255, 128, 128)
BLACK = (0, 128, 128)
RED = (128, 0, 255)
VIOLET = (128, 255, 255)
BLUE = (128, 255, 0)
GREEN = (128, 0, 0)'''

def valid_color(variance, square):
    #we accept a greater variance when we select more points, the color varies more on bigger area
    #we want less precision but not linear so we take log
    #we found good results with variance of color 140 for 10 points and luminance 1500 for 10 points
    var_color = int(60*np.log(square))
    var_lum = int(650*np.log(square))
    return(variance[0] < var_lum and variance[1]+variance[2] < var_color)

def square_distance(u,v):
    return (u[0]-v[0])**2 + (u[1]-v[1])**2

def min_distance(p, set):
    invalid_set = []
    for point in set:
        if square_distance(p, point) < 15000:
            invalid_set.append(point)
    return invalid_set

def valid_distance(bnw_img, point, set, color_mean):
    invalid_set = min_distance(point,set)
    #must be far from color
    for p in invalid_set:
        if (np.sum(np.absolute(bnw_img[p[0],p[1]] - color_mean)) < 30):
            return False
    return True
    
def draw_square(yuv_img, bnw_img, pixel, square, mask=None, desat=1.):
    assert 0. <= desat <= 1. 
    H,W,_ = yuv_img.shape
    hmin, hmax = max(pixel[0]-square,0), min(pixel[0]+square,H-1)
    wmin, wmax = max(pixel[1]-square,0), min(pixel[1]+square,W-1)

    points = [(random.randint(hmin, hmax),random.randint(wmin,wmax)) for _ in range(2*square+1)]
    esperance = []
    for p in points:
        esperance.append(yuv_img[p[0],p[1]]*desat) #(y , u and v)
    color_mean = np.mean(esperance, axis=0) + [0.,(1.- desat)*128.,(1.- desat)*128.]
    variance = np.var(esperance, axis=0)
    valid = valid_color(variance, square)
    #to increase damage in variance of isolated points, measure np.mean((y - np.mean(y))**3)

    if valid: #and valid_distance(bnw_img, pixel, set, color_mean)
        bnw_img[hmin:hmax+1,wmin:wmax+1,1:] = color_mean[1:]
        if mask is not None:
            mask[hmin:hmax+1,wmin:wmax+1] = 255
        return True
    return False

def create_source(target, NB_PTS = 0, MU = 10, S = 3, NB_ZONES = -1, desat = 1., withMask = False):
    yuv_img = cv.cvtColor(target, cv.COLOR_BGR2YUV)
    H, W, _ = yuv_img.shape
    
    bnw_img = np.copy(yuv_img)
    if withMask:
        mask = np.zeros((H,W,1))
    else:
        mask = None
    bnw_img[:,:,1:] = 128

    zones = 0
    for _ in range(NB_PTS):
        square = random.randint(MU-S, MU+S)
        pixel = (random.randint(5,H-4), random.randint(5,W-4))
        if bnw_img[pixel[0],pixel[1],2] == 128:
            drawn = draw_square(yuv_img, bnw_img, pixel, square, mask, desat)
            if drawn:
                zones+=1
            if zones == NB_ZONES:
                break

    source = cv.cvtColor(bnw_img, cv.COLOR_YUV2RGB)
    if withMask:
        source = np.append(source, mask, axis=2)
    return source.astype(np.uint8)


def create_chrom_or_lum(target, eps):
    yuv_img = cv.cvtColor(target, cv.COLOR_BGR2YUV)
    chrom_or_lum = np.copy(yuv_img)
    if random.random()<eps: #chrominance
        chrom_or_lum[:,:,0] = 128
    else:                   #luminance
        chrom_or_lum[:,:,1:] = 128
    source = cv.cvtColor(chrom_or_lum, cv.COLOR_YUV2RGB)
    return source


def deteriorate_image(target, blur=0, contrast=1., noise=0, motion_blur=0, compression_factor = 1):
    # Apply a slight Gaussian blur to simulate the lens imperfections
    if blur != 0:
        target = cv.GaussianBlur(target, (2*blur-1, 2*blur-1), 0)

    # Reduce saturation to make it look old
    if contrast != 1.:
        mean = np.ones(target.shape)
        target = contrast * target + (1 - contrast) * mean
        target = target.astype(np.uint8)

    # Add noise to simulate grain
    if noise!=0:
        gaussian_noise = np.random.normal(0, noise, target.shape)
        target = target + gaussian_noise
        target = np.clip(target, 0, 255).astype(np.uint8)

    # apply the motion blur
    if motion_blur != 0:
        motion_blur = 2*motion_blur + 1
        kernel_motion_blur = np.zeros((motion_blur, motion_blur))
        kernel_motion_blur[motion_blur//2, :] = np.ones(motion_blur)
        kernel_motion_blur = kernel_motion_blur / motion_blur
        target = cv.filter2D(target, -1, kernel_motion_blur)

    if compression_factor != 1:
        height, width = target.shape[:2]
        target = cv.resize(target, (width // compression_factor, height // compression_factor))
        target = cv.resize(target, (width, height), interpolation=cv.INTER_LINEAR)

    return target

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
#testing code

def test_deterioration():
    debut = time.time()
    blur=1 #max = 3
    desat=0.9 #min = 0.6
    noise=2 #max = 15
    motion_blur=4 #max = 15
    compression_factor = 1 #max = 2
    img = deteriorate_image(cv.imread('data/colorization/training/test/000000000144.jpg'), blur, desat, noise, motion_blur, compression_factor)
    cv.imwrite('output_deteriorated_image.jpg', img)

    print("time for 100000 images: ~",(time.time() - debut)*100000 / 60, "minutes")

def tests(path, out="result"):
    debut = time.time()

    path_out = path.split(".")[0]+"_"+out+".jpg"
    img1 = cv.imread(path)
    print(img1.dtype)
    #img_out = create_source(img1,NB_PTS=50, MU=120, S=50)
    img_out = create_source(img1)
    print(img_out)
    cv.imwrite(path_out,img_out)
    #cv.imwrite(path_out,cv.cvtColor(img_out,cv.COLOR_RGB2BGR))

    print("time for 100000 images: ~",(time.time() - debut)*100000 / 60, "minutes")

#tests("point.jpg")
#test_deterioration()