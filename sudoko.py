import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

class Game : 
    def __init__(self) -> None:
        self.d = [
            [0,6,0,0,0,0,0,0,5],
            [5,0,8,0,3,0,0,4,0],
            [0,2,0,0,0,8,0,0,0],
            [6,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,2,0],
            [3,0,5,0,0,9,0,0,1],
            [1,0,9,0,0,3,0,0,2],
            [0,8,0,0,0,0,0,0,0],
            [0,0,0,0,4,0,7,0,0]
        ]
    
    def initCheck(self) -> None:
        # number of apparnces 
        self.n = [0 for _ in range(9)]
        # 
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.sq = [[] for _ in range(9)]
        self.pr = [[[i for i in range(1,10)] for _ in range(9)] for _ in range(9)]


def extractimg(img  :cv2.Mat):
    # white color mask
    # img = cv2.imread(filein)
    # img = img[207:232 , 205 : 224]
    #converted = convert_hls(img)
    # image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    image = img
    # print(image)
    # lower = np.uint8([0, 200, 0])
    # upper = np.uint8([255, 255, 255])
    # white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([0, 0,   0])
    upper = np.uint8([200, 200, 200])
    yellow_mask = cv2.inRange(image, lower, upper)
    return yellow_mask
    

def getImg(img : cv2.Mat) : 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thr = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    return img_thr

def getLines(img_thr : cv2.Mat) : 
    # lines = cv2.HoughLinesP(img_thr, rho=1, theta=np.pi / 180, threshold=128, minLineLength=600, maxLineGap=30)
    lines = cv2.HoughLinesP(img_thr, rho=1, theta=np.pi / 180, threshold=0, minLineLength=600, maxLineGap=30)
    return lines 

def captureV():
    vid = cv2.VideoCapture(0)

    while(True):
    
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        # th_frame = getImg(frame)
        # lines= getLines(th_frame)
        # if lines is not None   : print(len(lines))
        # print(lines)

        pre_img = preprocess(frame)
        cont_img = contour(pre_img)

        # Display the resulting frame
        cv2.imshow('frame', cont_img)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

        
 
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()



def preprocess(iimg : cv2.Mat) : 
    img = iimg.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def contour(image : cv2.Mat) : 
    contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) == 0  : return image
    print(len(contours))
    res =  image.copy()
    cv2.drawContours(res, [contours[0]], -1, (0,255,0), 20)
    return res
    # for cnt in contours[:min(5,len(contours))]:
    #     #im = image.copy()
    #     #cv2.drawContours(im, cnt, -1, (255,255,255), 5)
    #     #self.show(im,'contour')
    #     if len(self.approx(cnt)) == 4:
    #         return cnt

# extractimg("sudoku.jpg")
captureV()