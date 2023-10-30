import tensorflow as tf
import numpy as np
import pydicom
from typing import Optional
from scipy.ndimage import measurements
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to non-interactive mode
import matplotlib.pyplot as plt
import SimpleITK as sitk
import requests
import os
import nibabel as nib
import cv2
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
app = Flask(__name__)
CORS(app)
from matplotlib.colors import ListedColormap

cmap_viridis = plt.cm.get_cmap('viridis')
colors = cmap_viridis(np.linspace(0.3, 1, 256))
colors[:50, 3] = 0  # Set transparency of the first 50 colors (purple) to 0
cmap_custom = ListedColormap(colors)
cmap_viridis = plt.cm.get_cmap('Purples')
colors = cmap_viridis(np.linspace(0.3, 1, 256))
colors[:50, 3] = 0  
cmap_custom2 = ListedColormap(colors)

def dice_coef(y_true, y_pred, smooth=1e-8):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  denom = K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f))
  dice = (2 * intersection + smooth) / (denom + smooth)
  return dice

def dice_loss(y_true, y_pred):
  return 1.0 - dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred):
  alpha = 0.55
  gamma = 2.
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  focal = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE,
                                              gamma=gamma,
                                              alpha=alpha)(y_true_f, y_pred_f)
  return K.sum(focal)


# Define a dictionary with custom objects
custom_objects = {
    'dice_coef': dice_coef,
    'dice_loss': dice_loss,
    'focal_loss': focal_loss,
    'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy
}

h5_file_path = 'csFinal.h5'  # Specify the path to the saved HDF5 model file
loaded_model = tf.keras.models.load_model(h5_file_path, custom_objects=custom_objects)
# loaded_model = tf.saved_model.load("./N")
def get_object_agatston(calc_object: np.ndarray, calc_pixel_count: int):
  object_max = np.max(calc_object)
  object_agatston = 0
  if 130 <= object_max < 200:
    object_agatston = calc_pixel_count * 1
  elif 200 <= object_max < 300:
    object_agatston = calc_pixel_count * 2
  elif 300 <= object_max < 400:
    object_agatston = calc_pixel_count * 3
  elif object_max >= 400:
    object_agatston = calc_pixel_count * 4
  return object_agatston
def compute_agatston_for_slice(ds, predicted_mask: Optional[np.ndarray],
                               min_calc_object_pixels = 3) -> int:
  def create_hu_image(ds):
    return ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
  if not predicted_mask is None:
    mask = predicted_mask
  else:
    mask = example.ground_truth_mask
  if np.sum(mask) == 0:
    return 0
  slice_agatston = 0
  pixel_volume = (ds.PixelSpacing[0] * ds.PixelSpacing[1])
  example=ds               
  hu_image = create_hu_image(example)
  labeled_mask, num_labels = measurements.label(mask,
                                                structure=np.ones((3, 3)))
  for calc_idx in range(1, num_labels + 1):
    label = np.zeros(mask.shape)
    label[labeled_mask == calc_idx] = 1
    calc_object = hu_image * label

    calc_pixel_count = np.sum(label)
    if calc_pixel_count <= min_calc_object_pixels:
      continue
    calc_volume = calc_pixel_count * pixel_volume
    object_agatston = round(get_object_agatston(calc_object, calc_volume))
    slice_agatston += object_agatston*ds.SliceThickness/3
  return round(slice_agatston,2)

def prepare_input_image(image, expand_dims):
    image = np.expand_dims(image, axis=0)
    if expand_dims:
        return np.expand_dims(image, axis=3)
    else:
        return np.expand_dims(image, axis=2)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list')
def list_files():
    return render_template('list.html')
@app.route('/project')
def project():
    return render_template('project.html')
@app.route('/us')
def us():
    return render_template('us.html')

@app.route('/calculate_agatston', methods=['POST'])
def calculate_agatston_score():
    try:
        file = request.files['dicom_file']
        ds = pydicom.dcmread(file)
        image_array = (ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept).astype(np.float32)
        predictions = loaded_model(prepare_input_image(image_array, True), training=False).numpy()
        predictions = np.squeeze(predictions)
        binarized_prediction = (predictions > 0.01).astype(np.float32)
        agatston_score = compute_agatston_for_slice(ds, binarized_prediction)

        plt.imshow(image_array, cmap='gray')
        plt.imshow(predictions, cmap='viridis', alpha=0.6)
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        seg_data_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.clf()
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        image_data_base64 = base64.b64encode(buffer.getvalue()).decode()


        return jsonify({"agatston_score": agatston_score, "image_data": image_data_base64,"seg_data":seg_data_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/calculate_agatstons', methods=['POST'])
def calculate_agatston_scores():
    try:
        uploaded_files = request.files.getlist('dicom_files')
        agatston_scores = []
        seg_images = []
        # Create a temporary directory to store the DICOM files
        os.mkdir('dcm')
        temp_dir = "./dcm"
        # Save the uploaded DICOM files to the temporary directory
        for file in uploaded_files:
              file_path = os.path.join(temp_dir, file.filename)
              file.save(file_path)
                # print(file_path)
      # Get DICOM file names in the temporary directory
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames("./dcm")

        # dicom_names = [os.path.join(temp_dir, filename) for filename in os.listdir(temp_dir) if filename.endswith('.dcm')]
        # dicom_names.sort()  # Sort the DICOM file names

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Added a call to PermuteAxes to change the axes of the data
        image = sitk.PermuteAxes(image, [2, 1, 0])


        api_url = "http://localhost:8000/infer/segmentation_unet_heart?session_id=123&output=image"

        # Create a temporary NIfTI file
        temp_nii_file_s = 'temp_study.nii.gz'
        with open(temp_nii_file_s, 'wb') as temp_file:
           
            nifti_file_path = temp_file.name
            sitk.WriteImage(image, nifti_file_path)
            print(nifti_file_path)
        # Send the POST request with the NIfTI file as 'file' in form-data
        with open(nifti_file_path, 'rb') as nifti_file:
            files = {'file': (os.path.basename(nifti_file_path), nifti_file, 'application/octet-stream')}
            response = requests.post(api_url, files=files)

        # Check the response
        if response.status_code == 200:    
            mask = response.content
            print("done")
        else:
            print(f"Request failed with status code {response.status_code}.")
            print(response.text)

        # Clean up the temporary NIfTI file
        os.remove(nifti_file_path)
        
        temp_nii_file = 'temp_mask.nii.gz'
        with open(temp_nii_file, 'wb') as temp_file:
            temp_file.write(mask)

        # Load the .nii.gz file
        nii_img = nib.load(temp_nii_file)

        # Get the image data as a 3D array
        img_data = nii_img.get_fdata()
        mask_data=[]
        for i in range(img_data.shape[0]):
          # Extract a 2D slice from the 3D volume
          kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
          # Perform dilation to fill in gaps
          filled_mask = cv2.dilate(img_data[i, :, :], kernel, iterations=1)
          mask_data.append(filled_mask)  # Use img_data.shape[2] to iterate over slices

        # Clean up the temporary NIfTI file
        os.remove(temp_nii_file)

        counter = 0
        for file in dicom_names:
            
            ds = pydicom.dcmread(file, force= True)
            image_array = (ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept).astype(np.float32)
            predictions = loaded_model(prepare_input_image(image_array, True), training=False).numpy()
            predictions = np.squeeze(predictions)
            binarized_prediction = (predictions > 0.03).astype(np.float32)

            mask_image = mask_data[counter]
            counter=counter+1          

            # Set pixels to 0 in the result image where the mask pixels are not equal to 1
            # Create an array of zeros with the same shape as mask_image
            merged_mask = np.zeros_like(mask_image)

            # Set values to 1 in merged_mask where mask_image is 1 and binarized_prediction is greater than 0.001
            merged_mask[mask_image == 1] = binarized_prediction[mask_image == 1]

            agatston_score = compute_agatston_for_slice(ds, merged_mask)

            if agatston_score > 0:
                # plt.imshow(mask_image, cmap='Blues', alpha=0.2)
                plt.figure(figsize=(10, 6))
                # Assuming you have your image and mask data already loaded into image_array, mask_image, and merged_mask
                plt.imshow(image_array, cmap='gray')
                plt.imshow(mask_image, cmap=cmap_custom2, alpha=0.3)

                plt.imshow(merged_mask, cmap=cmap_custom, alpha=0.99)  # Use the custom colormap here
                plt.axis('off')        
                # Overlay the non-zero pixels from mask_image with some transparency (alpha=0.4) in a different color
                buffer = BytesIO()
                plt.savefig(buffer, bbox_inches='tight', pad_inches=0, format='png')
                seg_data_base64 = base64.b64encode(buffer.getvalue()).decode()
                seg_images.append(seg_data_base64)

            agatston_scores.append(agatston_score)

        total_agatston_score = sum(agatston_scores)
        
        for file_path in dicom_names:
            os.remove(file_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            "total_agatston_score": total_agatston_score,
            "seg_images": seg_images
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
