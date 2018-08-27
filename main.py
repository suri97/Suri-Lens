import cv2
from Text_Detection import get_cord_img
from Text_Recognition import get_text_lbl
import argparse

def resize_img(image, newW, newH ):
    (H, W) = image.shape[:2]
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    return image, rW, rH

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", type=str,
	help="path to input image")

ap.add_argument("-east", "--east", type=str, default='./Assets/frozen_east_text_detection.pb',
	help="path to input EAST text detector")

ap.add_argument("-c", "--min-confidence", type=float, default=0.8,
	help="minimum probability required to inspect a region")

ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")

ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")

args = vars(ap.parse_args())

orig = cv2.imread( args["image"] )
image = orig.copy()
result = orig.copy()

image, rW, rH = resize_img(image, args["width"], args["height"])
(H, W) = image.shape[:2]
orig_cord = get_cord_img( args["east"], image, W, H, args["min_confidence"], rW, rH, result)
ans = get_text_lbl( orig_cord, result )
print (ans)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL )

cv2.imshow("Original", orig)
cv2.imshow( "Result", result )
cv2.waitKey(0)
cv2.destroyAllWindows()






