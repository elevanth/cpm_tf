import cv2
import imageio
import math
import numpy as np

def read_image(file, box_size):
    # input image [R, G, B]
    if isinstance(file, str):
        oriImg = imageio.imread(file)
    elif isinstance(file, imageio.core.util.Image):
        oriImg = file

    if oriImg is None:
        print('oriImg is None!')
        return None

    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

    scale = box_size / (oriImg.shape[0] * 1.0)
    scaled_img = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale,
                            interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((box_size, box_size, 3)) * 128

    if scaled_img.shape[1] < box_size:
        offset = scaled_img.shape[1] % 2
        output_img[:, int(box_size / 2 - math.ceil(scaled_img.shape[1] / 2)):int(
            box_size / 2 + math.ceil(scaled_img.shape[1] / 2) - offset), :] = scaled_img
    else:
        output_img = scaled_img[:, int(scaled_img.shape[1] / 2 - box_size / 2):int(
            scaled_img.shape[1] / 2 + box_size / 2), :]

    return output_img


# Compute gaussian kernel for input image
def gaussian_img(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

#
# def make_gaussian(size, fwhm=3, center=None):
#     """ Make a square gaussian kernel.
#     size is the length of a side of the square
#     fwhm is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """
#
#     x = np.arange(0, size, 1, float)
#     y = x[:, np.newaxis]
#
#     if center is None:
#         x0 = y0 = size // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]
#
#     return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)

def make_gaussian_batch(heatmaps, size, fwhm):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    stride = heatmaps.shape[1] // size

    batch_datum = np.zeros(shape=(heatmaps.shape[0], size, size, heatmaps.shape[3]))

    for data_num in range(heatmaps.shape[0]):
        for joint_num in range(heatmaps.shape[3] - 1):
            heatmap = heatmaps[data_num, :, :, joint_num]
            center = np.unravel_index(np.argmax(heatmap), (heatmap.shape[0], heatmap.shape[1]))

            x = np.arange(0, size, 1, float)
            y = x[:, np.newaxis]

            if center is None:
                x0 = y0 = size * stride // 2
            else:
                x0 = center[1]
                y0 = center[0]

            batch_datum[data_num, :, :, joint_num] = np.exp(
                -((x * stride - x0) ** 2 + (y * stride - y0) ** 2) / 2.0 / fwhm / fwhm)
        batch_datum[data_num, :, :, heatmaps.shape[3] - 1] = np.ones((size, size)) - np.amax(
            batch_datum[data_num, :, :, 0:heatmaps.shape[3] - 1], axis=2)

    return batch_datum

M_PI = 3.14159

def rad2Deg(rad):
    return rad * (180 / M_PI)


def deg2Rad(deg):
    return deg * (M_PI / 180)


def warpMatrix(sw, sh, theta, phi, gamma, scale, fovy):
    st = math.sin(deg2Rad(theta))
    ct = math.cos(deg2Rad(theta))
    sp = math.sin(deg2Rad(phi))
    cp = math.cos(deg2Rad(phi))
    sg = math.sin(deg2Rad(gamma))
    cg = math.cos(deg2Rad(gamma))

    halfFovy = fovy * 0.5
    d = math.hypot(sw, sh)
    sideLength = scale * d / math.cos(deg2Rad(halfFovy))
    h = d / (2.0 * math.sin(deg2Rad(halfFovy)))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    Rtheta = np.identity(4)
    Rphi = np.identity(4)
    Rgamma = np.identity(4)

    T = np.identity(4)
    P = np.zeros((4, 4))

    Rtheta[0, 0] = Rtheta[1, 1] = ct
    Rtheta[0, 1] = -st
    Rtheta[1, 0] = st

    Rphi[1, 1] = Rphi[2, 2] = cp
    Rphi[1, 2] = -sp
    Rphi[2, 1] = sp

    Rgamma[0, 0] = cg
    Rgamma[2, 2] = cg
    Rgamma[0, 2] = sg
    Rgamma[2, 0] = sg

    T[2, 3] = -h

    P[0, 0] = P[1, 1] = 1.0 / math.tan(deg2Rad(halfFovy))
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    F = np.matmul(Rtheta, Rgamma)
    F = np.matmul(Rphi, F)
    F = np.matmul(T, F)
    F = np.matmul(P, F)

    ptsIn = np.zeros(12)
    ptsOut = np.zeros(12)
    halfW = sw / 2
    halfH = sh / 2

    ptsIn[0] = -halfW
    ptsIn[1] = halfH
    ptsIn[3] = halfW
    ptsIn[4] = halfH
    ptsIn[6] = halfW
    ptsIn[7] = -halfH
    ptsIn[9] = -halfW
    ptsIn[10] = -halfH
    ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0

    ptsInMat = np.array([[ptsIn[0], ptsIn[1], ptsIn[2]], [ptsIn[3], ptsIn[4], ptsIn[5]], [ptsIn[6], ptsIn[7], ptsIn[8]],
                         [ptsIn[9], ptsIn[10], ptsIn[11]]], dtype=np.float32)
    ptsOutMat = np.array(
        [[ptsOut[0], ptsOut[1], ptsOut[2]], [ptsOut[3], ptsOut[4], ptsOut[5]], [ptsOut[6], ptsOut[7], ptsOut[8]],
         [ptsOut[9], ptsOut[10], ptsOut[11]]], dtype=np.float32)
    ptsInMat = np.array([ptsInMat])
    ptsOutMat = cv2.perspectiveTransform(ptsInMat, F)

    ptsInPt2f = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    ptsOutPt2f = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)

    i = 0

    while i < 4:
        ptsInPt2f[i][0] = ptsIn[i * 3 + 0] + halfW
        ptsInPt2f[i][1] = ptsIn[i * 3 + 1] + halfH
        ptsOutPt2f[i][0] = (ptsOutMat[0][i][0] + 1) * sideLength * 0.5
        ptsOutPt2f[i][1] = (ptsOutMat[0][i][1] + 1) * sideLength * 0.5
        i = i + 1

    M = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)
    return M


def warpImage(src, theta, phi, gamma, scale, fovy):
    halfFovy = fovy * 0.5
    d = math.hypot(src.shape[1], src.shape[0])
    sideLength = scale * d / math.cos(deg2Rad(halfFovy))
    sideLength = np.int32(sideLength)

    M = warpMatrix(src.shape[1], src.shape[0], theta, phi, gamma, scale, fovy)
    dst = cv2.warpPerspective(src, M, (sideLength, sideLength))
    mid_x = mid_y = dst.shape[0] // 2
    target_x = target_y = src.shape[0] // 2
    offset = (target_x % 2)

    if len(dst.shape) == 3:
        dst = dst[mid_y - target_y:mid_y + target_y + offset,
              mid_x - target_x:mid_x + target_x + offset,
              :]
    else:
        dst = dst[mid_y - target_y:mid_y + target_y + offset,
              mid_x - target_x:mid_x + target_x + offset]

    return dst
