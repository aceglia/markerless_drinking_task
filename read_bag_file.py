import pyrealsense2 as rs
import cv2
import os
import numpy as np

if __name__ == "__main__":
    bag_file = r"C:\Users\Usager\Documents\Amedeo\rgbd_ms_cime\videos\test_1\20260507_110437.bag"
    img_path = r"C:\Users\Usager\Documents\Amedeo\rgbd_ms_cime\videos\test_1\img"

    os.makedirs(img_path, exist_ok=True)

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object 
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, bag_file)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 60)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    colorizer = rs.colorizer()
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        frame_number = frames.frame_number
        # Get depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            continue

        depth_raw = np.asanyarray(depth_frame.get_data())

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        alpha, beta, gamma = 0.6, 0.4, 0
        blended_image = cv2.addWeighted(color_image, alpha, depth_color_image, beta, gamma)

        # Render image in opencv window
        # cv2.imshow("Depth Stream", blended_image)
        cv2.imwrite(os.path.join(img_path, f"color_{frame_number}.png"), color_image)
        cv2.imwrite(os.path.join(img_path, f"depth_{frame_number}.png"), depth_raw)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

