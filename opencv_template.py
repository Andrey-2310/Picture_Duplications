import cv2
from matplotlib import pyplot as plt


def get_good_points_len(matches, weight_koef, min_of_keypoints):
    good_points = []
    matches_mask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < weight_koef * n.distance:
            matches_mask[i] = [1, 0]
            good_points.append(m)

    return (good_points, matches_mask) \
        if len(good_points) < min_of_keypoints \
        else get_good_points_len(matches, weight_koef - 0.1, min_of_keypoints)


original = cv2.imread('./pictures/brandworkz/Brandworkz-Logo-3.png', 0)
comparable = cv2.imread('./pictures/brandworkz/Brandworkz-Logo-5.png', 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(original, None)
kp2, des2 = sift.detectAndCompute(comparable, None)

index_params = dict(algorithm=0, trees=5)  # algorithm = 0 is for KDTree
search_params = dict()

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

number_keypoints = len(kp1) if len(kp1) <= len(kp2) else len(kp2)
good_points, matches_mask = get_good_points_len(matches, 0.7, number_keypoints)

# draw_params = dict(matchColor=(0, 255, 0),
#                    singlePointColor=(255, 0, 0),
#                    matchesMask=matches_mask,
#                    flags=0)
#
# img3 = cv2.drawMatchesKnn(original, kp1, comparable, kp2, matches, None, **draw_params)
# plt.imshow(img3)
# plt.show()

print("Keypoints 1ST Image: " + str(len(kp1)))
print("Keypoints 2ND Image: " + str(len(kp2)))
print("GOOD Matches:", len(good_points))

# Define how similar they are
print("How good is the match: ", len(good_points) / number_keypoints * 100)
