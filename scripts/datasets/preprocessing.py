import os

import dlib
import numpy as np
from skimage import io
from skimage import transform as trans

def detect_and_align_faces(img, face_detector, lmk_predictor, template_path, template_scale=2, size_threshold=999):
    """
    Detects and align the faces in a image. Code from: https://github.com/chaofengc/Face-SPARNet.
    """
    align_out_size = (512, 512)
    ref_points = np.load(template_path) / template_scale
        
    # Detect landmark points
    face_dets = face_detector(img, 1)
    if len(face_dets) == 1: # Skip images with more than one face or no faces
        aligned_faces = []
        tform_params = []
        for det in face_dets:
            if isinstance(face_detector, dlib.cnn_face_detection_model_v1):
                rec = det.rect # for cnn detector
            else:
                rec = det
            if rec.width() > size_threshold or rec.height() > size_threshold: 
                print('Face is too large')
                break
            landmark_points = lmk_predictor(img, rec) 
            single_points = []
            for i in range(5):
                single_points.append([landmark_points.part(i).x, landmark_points.part(i).y])
            single_points = np.array(single_points)
            tform = trans.SimilarityTransform()
            tform.estimate(single_points, ref_points)
            tmp_face = trans.warp(img, tform.inverse, output_shape=align_out_size, order=3)
            aligned_faces.append(tmp_face*255)
            tform_params.append(tform)
        return [aligned_faces, tform_params]
    
    return [], []

def crop_and_align(img_path: str, out_path: str, out_img_name: str = ''):
    """
    Crops the faces from a given image. Code from: https://github.com/chaofengc/Face-SPARNet.

    The weights necessary for this code are defined in "weights.zip".
    """
    face_detector = dlib.cnn_face_detection_model_v1(".mmod_human_face_detector.dat")
    lmk_predictor = dlib.shape_predictor(".shape_predictor_5_face_landmarks.dat")
    template_path = ".FFHQ_template.npy"

    print('======> Loading images, crop and align faces.')
    img = dlib.load_rgb_image(img_path)
    aligned_faces, _ = detect_and_align_faces(img, face_detector, lmk_predictor, template_path)

    # Save aligned LQ faces
    img_name = out_img_name
    if out_img_name == '':
        img_name = img_path.split('/')[-1].split('.')[0]

    for idx, img in enumerate(aligned_faces):
        save_path = os.path.join(out_path, f"{img_name}_{idx}.png")
        try:
            io.imsave(save_path, img.astype(np.uint8))
        except Exception as e:
            raise e
        print("Saved", save_path)