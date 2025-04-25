import skimage.io as sio
import AFT_tools as AFT
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from PIL import Image, ImageQt
import cv2
import pandas as pd
import os
import io
from PyQt6.QtGui import QPixmap, QImage


def image_processing(image_paths):
    # AFT parameters
    #### required parameters ####
    window_size = 40
    overlap = 0.7
    neighborhood_radius = 5

    #### optional parameters ####
    intensity_threshold = 0
    eccentricity_threshold = 1
    #im_mask = io.imread('mask').astype('bool')

    #### output parameters ####
    plot_overlay = True
    plot_angles = False
    plot_eccentricity = False
    save_figures = True

    output_images = []

    N = len(image_paths)

    rotated_image = []
    
    angle_range = np.deg2rad(np.linspace(-80, 80, 9))
    Data = pd.DataFrame({'Angle': np.rad2deg(angle_range)})

    for i, path in enumerate(image_paths[:N]):  # Process only the first 3 images
        imag = sio.imread(path)
        if imag.shape[-1] == 4:
            imag = imag[..., :3]  # Keep only RGB
        imag = rgb2gray(imag)

        rot = rotate_image(image = imag, 
                            angle = 0,
                            crop_factor = 1,
                            off_axis_horizontal= 0,
                            off_axis_vertical = 0)
        rotated_image.append(rot)

        x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(imag, window_size, overlap,
                                                            plot_overlay=plot_overlay, plot_angles=plot_angles, 
                                                            plot_eccentricity=plot_eccentricity, save_figures=save_figures)
    
        Data[f'Imag {i + 1}'] = count_angles(im_theta = im_theta)

    mean = 0
    for column in Data.columns:
        if column == 'Angle':
            pass
        else:
            mean += Data[column]

    Data['Mean'] = mean/(len(Data.columns)-1)
    suma = Data['Mean'].sum()
    Data['Percentage'] = (Data['Mean']/suma)*100

    def render_plot_to_pixmap(plot_func):
        plt.clf()
        plot_func(angle_range, Data)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        img = Image.open(buf)
        qimg = ImageQt.toqpixmap(img)   # Convert PIL image to QPixmap
        return qimg

    img = Image.open("overlay_frame.tif")
    overlay_frame = ImageQt.toqpixmap(img)   # Convert PIL image to QPixmap
    os.remove("overlay_frame.tif")

    output_pixmaps = []
    output_pixmaps.append(render_plot_to_pixmap(polar_plot1))
    output_pixmaps.append(render_plot_to_pixmap(polar_plot2))
    output_pixmaps.append(render_plot_to_pixmap(polar_plot3))
    output_pixmaps.append(overlay_frame)

    return output_pixmaps

def polar_plot1(angle_range, Data):
    # Create polar plot
    plt.clf()
    ax = plt.subplot(projection='polar')

    # Plot bars for each range
    ax.set_rlabel_position(80)
    ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
    ax.grid(linewidth=0.7, alpha=0.6, zorder=1)  # Make grid lines fainter

    bars = ax.bar(angle_range, Data['Percentage'], width=np.pi / 9, edgecolor='b', linewidth=1, bottom=0.0, color='none', alpha=0.8)

    ax.set_yticks([x for x in ax.get_yticks() if x <= max(Data['Percentage'])])
    ax.set_ylim(0, max(Data['Percentage']))  # Set y-axis limit
    ax.set_yticklabels([f"{int(y)}%" for y in ax.get_yticks()])  # Format labels as percentages
    ax.spines['polar'].set_visible(False)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()): 
        item.set_fontsize(12)

def polar_plot2(angle_range, Data):
    # Create polar plot
    ax = plt.subplot(projection= 'polar')

    # Plot bars for each range
    ax.set_rlabel_position(80)
    ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
    ax.grid(linewidth=0.7, alpha=0.6, zorder=1)  # Make grid lines fainter

    bars = ax.bar(np.concatenate([angle_range, angle_range + np.pi]), np.concatenate([Data['Percentage'], Data['Percentage']]), width=np.pi / 9, edgecolor='b', linewidth=1, bottom=0.0, color='none', alpha=0.8)

    ax.set_yticks([x for x in ax.get_yticks() if x <= max(Data['Percentage'])])
    ax.set_ylim(0, max(Data['Percentage']))  # Set y-axis limit
    ax.set_yticklabels([f"{int(y)}%" for y in ax.get_yticks()])  # Format labels as percentages
    ax.spines['polar'].set_visible(False)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()): 
        item.set_fontsize(12)

def polar_plot3(angle_range, Data):
    # Create polar plot
    ax = plt.subplot(projection= 'polar')

    # Plot bars for each range
    ax.set_rlabel_position(80)
    ax.grid(linewidth=0.7, alpha=0.6, zorder=1)  # Make grid lines fainter

    bars = ax.bar(angle_range, Data['Percentage'], width=np.pi / 9, edgecolor='b', linewidth=1, bottom=0.0, color='none', alpha=0.8)

    ax.set_ylim(0, max(Data['Percentage']))  # Set y-axis limit
    ax.set_yticks([x for x in ax.get_yticks() if x <= max(Data['Percentage'])])
    ax.set_yticklabels([f"{int(y)}%" for y in ax.get_yticks()])  # Format labels as percentages
    ax.spines['polar'].set_visible(False)

    # Set 0 in the center
    ax.set_theta_zero_location('N')
    ax.set_thetalim(-np.pi/2, np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()): 
        item.set_fontsize(12)

def rotate_image(image, angle, crop_factor, off_axis_horizontal = 0, off_axis_vertical = 0):
    """
    Rotates the given image by the specified angle.
    
    Parameters:
        image: numpy.ndarray
            The input image.
        angle: float
            The angle of rotation in degrees.
    
    Returns:
        numpy.ndarray
            The rotated image.
    """
    # Get the height and width of the image
    height, width = image.shape[:2]
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    
    # Perform the affine transformation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    rotated_height, rotated_width = rotated_image.shape[:2]
    
    crop_height = int(rotated_height * crop_factor)
    crop_width = int(rotated_width * crop_factor)
    
    # Calculate the position to crop the central piece
    crop_x = (rotated_width - crop_width) // 2 + off_axis_horizontal
    crop_y = (rotated_height - crop_height) // 2 + off_axis_vertical
    
    # Crop the central piece from the rotated image
    cropped_image = rotated_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    
    return cropped_image

def count_angles(im_theta):
    # Sample data (angles in degrees)
    angles_deg = im_theta.reshape((im_theta.shape[0]*im_theta.shape[1],1)) * 180 / np.pi

    # Define angle ranges
    angle_ranges = [(-90, -70), (-70, -50), (-50, -30), (-30, -10), (-10, 10), (10, 30), (30, 50), (50, 70), (70, 90)]
    #angle_ranges = [(0, 15), (15, 30), (30, 45), (45,60), (60,75), (75,90)]

    # Initialize counts for each range
    counts = np.zeros(len(angle_ranges), dtype=int)

    # Count vectors in each range
    for i, (start, end) in enumerate(angle_ranges):
        counts[i] = np.sum((angles_deg >= start) & (angles_deg < end))

    # Print counts
    for i, (start, end) in enumerate(angle_ranges):
        print(f'Number of vectors in range {start} - {end}: {counts[i]}')
        
    return counts


if __name__ == "__main__":
    print("AA")