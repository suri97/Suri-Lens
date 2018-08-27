import pytesseract
import cv2

def process_img(img):
    img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    img = cv2.threshold(img,210,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    img = cv2.GaussianBlur(img, (3 , 3), 0)
    return img

def put_label(img, lbl, cord):
    startX, startY, endX, endY = cord
    cv2.rectangle(img, (startX-1, startY), (endX - 1, startY-20), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, lbl, (startX+5, startY-5), font, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

def get_text_lbl(orig_cord, image):
    ans = ""

    for i in orig_cord:
        startX, startY, endX, endY = i

        req_image = image[startY:endY, startX:endX].copy()
        req_image = process_img(req_image)

        text = pytesseract.image_to_string(req_image)

        if (len(text) > 0):
            ans += text + '\n'
            put_label(image, text, i)

    return ans
