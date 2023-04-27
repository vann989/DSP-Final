import cv2
from matplotlib import pyplot as plt
import numpy as np

class DFT():
    def __init__(self, img):
        self.grayImg = cv2.imread(img,0)
        self.img = cv2.imread(img)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)

    def HPF(self,img,r):
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0


        # Apply mask and inverse DFT
        fshift = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)) * mask
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return img_back
    
    def LPF(self,img,r):
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        # Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r_out = 80
        r_in = 10
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                                ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
        mask[mask_area] = 1


        # apply mask and inverse DFT
        fshift = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)) * mask

        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return img_back
    def colorHPF(self,r):
        (R,G,B) = cv2.split(self.img)
        R = self.HPF(R,r)
        G = self.HPF(G,r)
        B = self.HPF(B,r)
        return cv2.merge([R/500000000,G/500000000,B/500000000])

    def colorLPF(self,r):
        (R,G,B) = cv2.split(self.img)
        R = self.LPF(R,r)
        G = self.LPF(G,r)
        B = self.LPF(B,r)
        return cv2.merge([R/500000000,G/500000000,B/500000000])

    def showHPFTransform(self,r):
        plt.imshow(self.HPF(self.grayImg,r))
        plt.show()
    def showLPFTransform(self,r):
        plt.imshow(self.LPF(self.grayImg,r))
        plt.show()
    def showColorHPFTransform(self,lowR,highR):
        f = plt.figure(figsize=(12, 12))
        inc = (highR-lowR)/10
        for i in range(0,10):
            f.add_subplot(2,5,i+1)
            plt.imshow(self.colorHPF(inc*i))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title(inc*i)
        plt.show()

    def showColorLPFTransform(self,lowR,highR):
        f = plt.figure(figsize=(12, 12))
        inc = (highR-lowR)/10
        for i in range(0,10):
            f.add_subplot(2,5,i+1)
            plt.imshow(self.colorLPF(inc*i))
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title(inc*i)
        plt.show()

    def showImg(self):
        plt.imshow(self.img)
        plt.show()



Bonk = DFT('Bonk.jpeg')

#Bonk.showImg()
#Bonk.showHPFTransform(10)
#Bonk.showLPFTransform(1)
Bonk.showColorLPFTransform(0,10)