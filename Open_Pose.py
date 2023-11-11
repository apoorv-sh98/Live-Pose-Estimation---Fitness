import cv2
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from deep_pose.model import create_deep_pose
# from deep_pose.utils import draw_pose
# import posenet
# Load PoseNet model
# model = 101
# scale_factor = 0.4
# with tf.Session() as sess:
#     # Load PoseNet model
#     model_cfg, model_outputs = posenet.load_model(model, sess)
#     output_stride = model_cfg['output_stride']
    # start = time.time()

#####
### Uncomment below function for setting model paramters
#####

# def set_params():

#         params = dict()
#         params["logging_level"] = 3
#         params["output_resolution"] = "-1x-1"
#         params["net_resolution"] = "-1x368"
#         params["model_pose"] = "BODY_25"
#         params["alpha_pose"] = 0.6
#         params["scale_gap"] = 0.3
#         params["scale_number"] = 1
#         params["render_threshold"] = 0.05
#         # If GPU version is built, and multiple GPUs are available, set the ID here
#         params["num_gpu_start"] = 0
#         params["disable_blending"] = False
#         # Ensure you point to the correct path where models are located
#         params["default_model_folder"] = dir_path + "/../../../models/"
#         return params




def main():


    # params = set_params()

        #Constructing OpenPose object allocates GPU memory
    # openpose = OpenPose(params)
    capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
    # Capture frame-by-frame
        ret, frame = capture.read()

    # # Display the resulting frame
    # cv2.imshow('Webcam Stream', frame)
        if frame is not None:
        # run the model on the image and generate output results
        # try:
        #     input_image, draw_image, output_scale = posenet.read_cap(
        #         frame, scale_factor=scale_factor, output_stride=output_stride)
        #     heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        #         model_outputs,
        #         feed_dict={'image:0': input_image}
        #     )
        # except Exception as e:
        #     print(f"Error processing frame: {e}")
        #     break


        #-------------------------------------------------------------------#
        # For openpose
        #-------------------------------------------------------------------#
            # Display the stream
            # keypoints, output_image = openpose.forward(img, True)

            #         # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
            #         if len(keypoints)>0:
            #                 print('Human(s) Pose Estimated!')
            #                 print(keypoints)
            #         else:
            #                 print('No humans detected!')
            cv2.putText(frame,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.imshow('Webcam Stream', frame) # change frame-> output_image when open is running
        # cv2.imshow('Webcam width', width)q
        # cv2.imshow('Webcam height', height)
        # cv2.imshow('Webcam fpd', fps)

    # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
            key = cv2.waitKey(1)

            if key==ord('q'):
                break

    # Release the webcam and close all windows
    capture.release()
    cv2.destroyAllWindows()