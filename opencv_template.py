import cv2
from matplotlib import pyplot as plt

original = cv2.imread('./pictures/coca-cola-default.jpg', 0)
image_to_compare = cv2.imread('./pictures/coca-cola-180.jpg', 0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(original, None)
kp2, des2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)  # algorithm = 0 is for KDTree
search_params = dict()

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
good_points = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
        good_points.append(m)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(original, kp1, image_to_compare, kp2, matches, None, **draw_params)
plt.imshow(img3, ), plt.show()

# Define how similar they are
number_keypoints = len(kp1) if len(kp1) <= len(kp2) else len(kp2)
print("Keypoints 1ST Image: " + str(len(kp1)))
print("Keypoints 2ND Image: " + str(len(kp2)))
print("GOOD Matches:", len(good_points))
print("How good it's the match: ", len(good_points) / number_keypoints * 100)
