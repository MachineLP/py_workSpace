
import cv2
import numpy as np

############################################################################
# 函数：crop
# 描述：随机裁剪图像
#
# 输入：图像image, crop_size
# 返回：图像image
############################################################################
def crop(image, crop_size, random_crop=True):
    if random_crop:  # 若随机裁剪
        if image.shape[1] > crop_size:
            sz1 = image.shape[1] // 2
            sz2 = crop_size // 2
            diff = sz1 - sz2
            (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
            image = image[v:(v + crop_size), h:(h + crop_size), :]

    return image

############################################################################
# 函数：flip
# 描述：随机反转图片，增强数据
#
# 输入：图像image
# 返回：图像image
############################################################################
def flip(image, random_flip=True):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

############################################################################
# 函数：rotation
# 描述：随机旋转图片，增强数据，用图像边缘进行填充。
#
# 输入：图像image
# 返回：图像image
############################################################################
def rotation(image, random_flip=True):
    if random_flip and np.random.choice([True, False]):
        w,h = image.shape[1], image.shape[0]
        # 0-180随机产生旋转角度。
        angle = np.random.randint(0,180)
        RotateMatrix = cv2.getRotationMatrix2D(center=(image.shape[1]/2, image.shape[0]/2), angle=angle, scale=0.7)
        # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
        #image = cv2.warpAffine(image, RotateMatrix, (w,h),borderValue=(129,137,130))
        image = cv2.warpAffine(image, RotateMatrix, (w,h),borderMode=cv2.BORDER_REPLICATE)
    return image


############################################################################
# 函数：translation
# 描述：随机平移图片，增强数据，用图像边缘进行填充。
#
# 输入：图像image
# 返回：图像image
############################################################################
def translation(image, random_flip=True):
    if random_flip and np.random.choice([True, False]):
        w,h = 1920, 1080
        H1 = np.float32([[1,0],[0,1]])
        H2 = np.random.uniform(50,500, [2,1])
        H = np.hstack([H1, H2])
        # H = np.float32([[1,0,408],[0,1,431]])
        print (H)
        image = cv2.warpAffine(image, H, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return image




# 图像的平移。
img = cv2.imread('lp_aug.jpg')
print (img[20][20])
# 平移参数。
H = np.float32([[1,0,500],[0,1,50]])
# RotateMatrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=0, scale=0.5)
# h,w = img.shape[:2]
w,h = 1920, 1080
RotImg = cv2.warpAffine(img, H, (w,h), borderMode=cv2.BORDER_REPLICATE) #需要图像、变换矩阵、变换后的大小
cv2.imwrite("out.jpg", RotImg)

# w,h = 1920, 1080
w,h = 1920, 1080
w,h = img.shape[1], img.shape[0]
# 旋转参数。 angle图像的旋转角度，scale图像的缩放比例。 scale = 0.5
RotateMatrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=45, scale=0.7)
# RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0]*2, img.shape[1]*2))
RotImg = cv2.warpAffine(img, RotateMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite("out_rot.jpg", RotImg)


img_r = rotation(img, True)
cv2.imwrite("out_rot1.jpg", img_r)

img_t = translation(img, True)
cv2.imwrite("out_1.jpg", img_t)



