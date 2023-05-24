import math as m
import os
import os.path as p
import random
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt



def get_bpp(image):
    h, w = image.shape
    image_list = image.tolist()
    bits = 0
    for row in image_list:
        for pixel in row:
            bits += m.log2(abs(pixel) + 1)
    return bits / (h * w)

def get_image_frames(file, first, second):
    cap = c.VideoCapture(file)
    cap.set(c.CAP_PROP_POS_FRAMES, first - 1)
    res1, fr1 = cap.read()
    cap.set(c.CAP_PROP_POS_FRAMES, second - 1)
    res2, fr2 = cap.read()
    cap.release()
    return fr1, fr2

def get_reconstructed_target(residual, predicted):
    return np.add(residual, predicted)


def get_residual(target, predicted):
    return np.subtract(target, predicted)


def block_search_body(anchor, target, block_size, search_area=7):
    h, w = anchor.shape
    h_segments, w_segments = segment_image(anchor, block_size)
    predicted = np.ones((h, w)) * 255
    b_count = 0
    for y in range(0, int(h_segments * block_size), block_size):
        for x in range(0, int(w_segments * block_size), block_size):
            b_count += 1
            target_block = target[y:y + block_size, x:x + block_size]
            anchor_search_area = get_anchor_search_area(x, y, anchor, block_size, search_area)
            anchor_block = get_best_match(target_block, anchor_search_area, block_size)
            predicted[y:y + block_size, x:x + block_size] = anchor_block
    assert b_count == int(h_segments * w_segments)
    return predicted


def segment_image(anchor, block_size=16):
    h, w = anchor.shape
    h_segments = int(h / block_size)
    w_segments = int(w / block_size)
    return h_segments, w_segments


def get_anchor_search_area(x, y, anchor, block_size, search_area):
    h, w = anchor.shape
    cx, cy = get_center(x, y, block_size)
    sx = max(0, cx - int(block_size / 2) - search_area)
    sy = max(0, cy - int(block_size / 2) - search_area)
    anchor_search = anchor[sy:min(sy + search_area * 2 + block_size, h),
                     sx:min(sx + search_area * 2 + block_size, w)]
    return anchor_search


def get_center(x, y, block_size):
    return int(x + block_size / 2), int(y + block_size / 2)


def get_best_match(t_block, a_search, block_size):
    step = 4
    ah, aw = a_search.shape
    acy, acx = int(ah / 2), int(aw / 2)
    min_mad = float("+inf")
    min_p = None
    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx + step, acy)
        p3 = (acx, acy + step)
        p4 = (acx + step, acy + step)
        p5 = (acx - step, acy)
        p6 = (acx, acy - step)
        p7 = (acx - step, acy - step)
        p8 = (acx + step, acy - step)
        p9 = (acx - step, acy + step)
        point_list = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        for p in range(len(point_list)):
            a_block = get_block_zone(point_list[p], a_search, t_block, block_size)
            mad = get_mad(t_block, a_block)
            if mad < min_mad:
                min_mad = mad
                min_p = point_list[p]
        step = int(step / 2)
    px, py = min_p
    px, py = px - int(block_size / 2), py - int(block_size / 2)
    px, py = max(0, px), max(0, py)
    match_block = a_search[py:py + block_size, px:px + block_size]
    return match_block


def get_block_zone(p, a_search, t_block, block_size):
    px, py = p
    px, py = px - int(block_size / 2), py - int(block_size / 2)
    px, py = max(0, px), max(0, py)
    a_block = a_search[py:py + block_size, px:px + block_size]
    try:
        assert a_block.shape == t_block.shape
    except Exception as e:
        print(e)
    return a_block


def get_mad(t_block, a_block):
    return np.sum(np.abs(np.subtract(t_block, a_block))) / (t_block.shape[0] * t_block.shape[1])


def main(anchor_frame, target_frame, block_size, save_output=True):
    bpp_anchor = []
    bpp_diff = []
    bpp_predicted = []
    h, w, ch = anchor_frame.shape
    print(h, w, ch)
    diff_frame_rgb = np.zeros((h, w, ch))
    predicted_frame_rgb = np.zeros((h, w, ch))
    residual_frame_rgb = np.zeros((h, w, ch))
    restore_frame_rgb = np.zeros((h, w, ch))
    for i in range(0, 3):
        anchor_frame_c = anchor_frame[:, :, i]
        target_frame_c = target_frame[:, :, i]
        diff_frame = c.absdiff(anchor_frame_c, target_frame_c)
        predicted_frame = block_search_body(anchor_frame_c, target_frame_c, block_size)
        residual_frame = get_residual(target_frame_c, predicted_frame)
        reconstruct_target_frame = get_reconstructed_target(residual_frame, predicted_frame)
        bpp_anchor += [get_bpp(anchor_frame_c)]
        bpp_diff += [get_bpp(diff_frame)]
        bpp_predicted += [get_bpp(residual_frame)]
        diff_frame_rgb[:, :, i] = diff_frame
        predicted_frame_rgb[:, :, i] = predicted_frame
        residual_frame_rgb[:, :, i] = residual_frame
        restore_frame_rgb[:, :, i] = reconstruct_target_frame
    output_dir = "Results"
    is_dir = p.isdir(output_dir)
    if not is_dir:
        os.mkdir(output_dir)
    if save_output:
        c.imwrite(f"{output_dir}/First_frame.png", anchor_frame)
        c.imwrite(f"{output_dir}/Second_frame.png", target_frame)
        c.imwrite(f"{output_dir}/Difference_between_frames.png", diff_frame_rgb)
        c.imwrite(f"{output_dir}/Prediction_frame.png", predicted_frame_rgb)
        c.imwrite(f"{output_dir}/Residual_frame.png", residual_frame_rgb)
        c.imwrite(f"{output_dir}/Restore_frame.png", restore_frame_rgb)
        bar_width = 0.25
        fig = plt.subplots(figsize=(12, 8))
        p1 = [sum(bpp_anchor), bpp_anchor[0], bpp_anchor[1], bpp_anchor[2]]
        diff = [sum(bpp_diff), bpp_diff[0], bpp_diff[1], bpp_diff[2]]
        mpeg = [sum(bpp_predicted), bpp_predicted[0], bpp_predicted[1], bpp_predicted[2]]
        br1 = np.arange(len(p1))
        br2 = [x + bar_width for x in br1]
        br3 = [x + bar_width for x in br2]
        br4 = [x + bar_width for x in br3]
        plt.bar(br1, p1, color='r', width=bar_width, edgecolor='grey', label='Bit per pixel for anchor frame')
        plt.bar(br2, diff, color='g', width=bar_width, edgecolor='grey',
                label='Bit per pixel for difference between frames')
        plt.bar(br3, mpeg, color='b', width=bar_width, edgecolor='grey',
                label='Bit per pixel for motion compensated difference')
        plt.title(f'Compression ratio = {round(sum(bpp_anchor) / sum(bpp_predicted), 2)}', fontweight='bold', fontsize=15)
        plt.ylabel('Bit per pixel', fontweight='bold', fontsize=15)
        plt.xticks([r + bar_width for r in range(len(p1))],
                   ['Bit/Pixel RGB', 'Bit/Pixel R', 'Bit/Pixel G', 'Bit/PixelB'])
        plt.legend()
        plt.savefig(f'{output_dir}/Bit_per_pixel_histogram_for_different_encodings.png', dpi=400)


if __name__ == "__main__":
    f = random.randint(0, 3000)
    fr1, fr2 = get_image_frames('video.avi', f, f + 1)
    main(fr1, fr2, 32, save_output=True)
