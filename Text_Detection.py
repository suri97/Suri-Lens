import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import sys

def load_model(model_path):
    print ('[INFO] Loading EAST Text Detector ... ')
    try:
        net = cv2.dnn.readNet(model_path)
        return net
    except:
        print("Model Path isn't correct")
        sys.exit(1)


def frwrd_pass(net, image, W, H):
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    print ('Preparing for Forward Pass')
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    return scores, geometry

def get_rect( scores, geometry, conf ):

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < conf:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences

def get_cord_img( model_path, image, W, H, conf, rW, rH, for_text ):
    model = load_model(model_path)
    scores, geometry = frwrd_pass(model, image, W, H)
    rects, confidences = get_rect(scores, geometry, conf)

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    orig_cord = []

    for (startX, startY, endX, endY) in boxes:

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        orig_cord.append((startX, startY, endX, endY))

    orig_cord = sorted(orig_cord, key=lambda x: x[1])

    return orig_cord

