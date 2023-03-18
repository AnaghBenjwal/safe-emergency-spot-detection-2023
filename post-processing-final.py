import cv2
import numpy as np

image_original = cv2.imread(r"uavid_v1.5_official_release_image_bw\uavid_patch_train\Images\seq14_000200_0_5.png")
img_pred = cv2.imread(r"uavid_v1.5_official_release_image_bw\uavid_patch_train\Masks\seq14_000200_0_5.png")

# image = cv2.resize(img_pred, (1024, 1024))
imgray = cv2.cvtColor(img_pred, cv2.COLOR_RGB2GRAY)
imgmorph = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

_, imgbin = cv2.threshold(imgmorph, 0, 255, cv2.THRESH_BINARY)  #Binary Image
# imgbin = cv2.bitwise_not(imgbin)
imgdist = cv2.distanceTransform(imgbin, cv2.DIST_L2, 5, cv2.DIST_MASK_PRECISE)

_, max_val, _, centre = cv2.minMaxLoc(imgdist)
print(max_val)

if max_val > 50 :
    cv2.imwrite('PaperPictures/igm1_original.jpg', image_original)
    cv2.imwrite('PaperPictures/img2_prediction.jpg', img_pred)
    cv2.imwrite('PaperPictures/img3_morphopening.jpg', imgmorph)
    cv2.imwrite('PaperPictures/img4_DistTransformed.jpg', imgdist)

    circle = cv2.circle(image_original, centre, int(max_val), (0, 0, 255), 2)
    circle = cv2.circle(img_pred, centre, int(max_val), (0, 0, 255), 2)

    cv2.imwrite('PaperPictures/img5_predwithcircle.jpg', img_pred)
    cv2.imwrite('PaperPictures/img6_imgwithcircle.jpg', image_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else : print("identified circle did not pass threshold requirement")



