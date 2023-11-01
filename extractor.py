import os
import shutil

import cv2


# rescale the images
def rescale(img):
    scale = 0.5
    h, w = img.shape[:2]
    h = int(h * scale)
    w = int(w * scale)
    return cv2.resize(img, (w, h))


# delete and create directory
folder = "frames/"
if os.path.isdir(folder):
    shutil.rmtree(folder)
os.mkdir(folder)

# open vidcap
cap = cv2.VideoCapture("videos\\VID_20231101_164150.mp4")  # your video here
counter = 0

# make an orb feature detector and a brute force matcher
orb = cv2.ORB.create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# store the first frame
_, last = cap.read()
last = rescale(last)
cv2.imwrite(folder + str(counter).zfill(5) + ".png", last)

# get the first frame's stuff
kp1, des1 = orb.detectAndCompute(last, None)

# cutoff, the minimum number of keypoints
cutoff = 50
# Note: this should be tailored to your video, this is high here since a lot of this video looks like

# count number of frames
prev = None
while True:
    # get frame
    ret, frame = cap.read()
    if not ret:
        break

    # resize
    frame = rescale(frame)

    # count keypoints
    kp2, des2 = orb.detectAndCompute(frame, None)

    # match
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.5 * n.distance]
    # check against cutoff
    print(len(good))
    if len(good) < cutoff:
        # swap and save
        counter += 1
        last = frame
        kp1 = kp2
        des1 = des2
        cv2.imwrite(folder + str(counter).zfill(5) + ".png", last)
        print(f"New Frame: {counter}")

    # show
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    prev = frame

# also save last frame
counter += 1
cv2.imwrite(folder + str(counter).zfill(5) + ".png", prev)

# check number of saved frames
print(f"Counter: {counter}")
