import numpy as np
import cv2
import sys
import os
from glob import glob
def click_event(event, x, y, flags, param):
    coors = param[0]
    img = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        coors.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

def main(filename):
    img = cv2.imread(filename, 1)
    h, w = img.shape[:2]
    f = 4
    h = int(h/f)
    w = int(w/f)
    # along x-axis and then along y-axis
    img = cv2.resize(img, (w, h))
    cv2.imshow('image', img)
    coors = []
    cv2.setMouseCallback('image', click_event, [coors, img])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(coors)
    if len(coors) != 0:
        # fix order: left-top left-bottom right-bottom right-top
        coors = np.array(coors)*f
        w = int(np.sqrt(np.sum((coors[-1] - coors[0])**2)))
        h = int(np.sqrt(np.sum((coors[1] - coors[0])**2)))
        output = filename.replace(".jpg", "_persp.jpg")
        cmd = f'magick {filename} -distort perspective "{coors[0][0]},{coors[0][1]},0,0 {coors[1][0]},{coors[1][1]},0,{h} {coors[2][0]},{coors[2][1]},{w},{h} {coors[3][0]},{coors[3][1]},{w},0" -crop {w}x{h}+0+0 {output}'
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    # if "*" not in sys.argv[1]:
    #     # main(sys.argv[1])
    #     print(str(sys.argv[1]))
    #     print(sys.argv[1:-1])
    # else:
    #     imgs = glob(sys.argv[1])
    #     print(imgs)
    # print(len(sys.argv))
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) > 2:
        imgs = sys.argv[1:]
        for img in imgs:
            if "_persp.jpg" in img:
                continue
            elif img.replace(".jpg", "_persp.jpg") in imgs:
                continue
            else:
                main(img)
