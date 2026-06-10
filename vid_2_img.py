import cv2
import os

import numpy as np

if __name__ == '__main__':
#     video_dir = r'D:\Documents\Programmation\pose-estimation-finetune\annotations\data\calibration_2'
#     video_path = os.listdir(video_dir)
#     video_path = [os.path.join(video_dir, file) for file in video_path if file.endswith('.mp4')][0]
    video_path = r"C:\Users\Usager\Documents\Amedeo\rgbd_ms_cime\videos\P2\GX010093_crop.mp4"
    images_output_dir = os.path.join(os.path.dirname(video_path), 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_frame_time = 0
    n_interval = 1

    while True:
        ret, frame = cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            break
        # cur_frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        # if frame_count > 0:
        #     n_interval = np.round((cur_frame_ts - prev_frame_time) / ((1/30) * 1000), 0).astype(int)
        #     prev_frame_time = cur_frame_ts
        #     frame_count += max(0, (n_interval - 1))

        cv2.imwrite(os.path.join(images_output_dir, f'color_{frame_count}.png'), frame)
        frame_count += 1


        # img = frame

        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # h, s, v = cv2.split(hsv)

        # # reflection mask
        # mask = (v > 220) & (s < 30)
        # mask = mask.astype(np.uint8) * 255

        # b,g,r = cv2.split(img)

        # reflection = np.abs(r-g) + np.abs(r-b) + np.abs(g-b)
        # ret, test = cv2.threshold(
        #         img, int(50), int(200), 0
        #     )
        # mask = reflection < 70
        # mask = mask.astype(np.uint8)*255
        # np.concatenate(([mask[..., None]] * 3), axis=-1)
        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.imshow("mask", mask)
        # cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
        # cv2.imshow("raw", img)
        # cv2.waitKey(0)
        # frame_count += 1
    cap.release()

    