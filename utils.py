import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize


# display 1 greyscale image
def display_img(img):
	plt.figure(figsize=(14, 8))
	plt.imshow(img, cmap='gray', vmin=0, vmax=255)
	plt.show()


# display colorimage
def display_colorimg(img):
	plt.figure(figsize=(14, 8))
	plt.imshow(img)
	plt.show()


# display 2-4 images side by side GRAYSCALE
def display_sbs(*images, titles=None):
	num_images = len(images)
	if num_images < 2 or num_images > 4:
		raise ValueError('Please provide between 2 and 4 images.')

	ncols = 2
	nrows = (num_images + 1) // 2

	fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
	axes = (
		axes.flatten()
		if num_images > 2
		else axes
		if isinstance(axes, (list, np.ndarray))
		else [axes]
	)

	for i in range(len(axes)):
		if i < num_images:
			axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=255)
			axes[i].axis('off')
			if titles and i < len(titles):
				axes[i].set_title(titles[i])
		else:
			axes[i].axis('off')

	plt.tight_layout()
	plt.show()


def show_image_with_grayscale_channels(image, channels, threshold=0.5):
	# Normalize image and mask to [0, 1]
	image = image.astype(np.float32)
	if image.max() > 1.0:
		image /= 255.0
	image = 1.0 - image

	channels = channels.astype(np.float32)
	if channels.max() > 1.0:
		channels /= 255.0

	channel_mask = channels < threshold

	rgb_image = np.stack([image] * 3, axis=-1)
	overlay = rgb_image.copy()
	overlay[channel_mask] = [1, 0, 0]

	alpha = 0.5
	blended = rgb_image.copy()
	blended[channel_mask] = (1 - alpha) * rgb_image[channel_mask] + alpha * overlay[
		channel_mask
	]

	# Display
	plt.figure(figsize=(12, 8))
	plt.imshow(blended)
	plt.axis('off')
	plt.title('Channel Overlay on Grayscale Image')
	plt.show()


def skeletonize_img(image):
	# image must be white on black background
	# returns white on black background
	binary = (image > 0).astype(np.uint8)
	image = skeletonize(binary)
	image = (image * 255).astype(np.uint8)
	return image


def count_moves(arr):
	return np.argmax(arr != arr[0]) if np.any(arr != arr[0]) else 0


def get_idx(length):
	# length of slice
	if length % 2 == 0:
		return np.array((length // 2 - 1, length // 2))  # even
	else:
		return np.array((length // 2,))  # odd


def extractor(img, d=5):
	# extracts centreline of a detection (from left and right channel banks)
	# img ... edge image, white on black
	# d ... max distance between left and right bank
	# NOTE: channels must run vertically in the image

	img = img / np.max(img)

	center = np.zeros(img.shape)
	visited = np.full(img.shape, False)

	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			if visited[row, col]:
				continue
			visited[row, col] = True
			if img[row, col] == 0:
				continue
			m_left = count_moves(img[row, col:])
			m_space = count_moves(img[row, (col + m_left) : (col + m_left + d)])

			m_right, space_visited = 0, 0
			if m_space != 0:
				post_space = col + m_left + m_space
				m_right = count_moves(img[row, post_space:])
			else:
				space_visited += d

			visited[row, col : (col + m_left + m_space + m_right + space_visited)] = (
				True
			)
			idx = col + get_idx(img[col : (col + m_left + m_space + m_right)].shape[0])
			center[row, idx] = 1

	center = (center * 255).astype(np.uint8)

	return center


def detection_metrics(detection, truth, base_image=None):
	# visualize and evaluate channel detection results against a ground truth mask
	# detections are white (255) on black(0) background
	# returns metrics and an RGB image:
	# green = matched edge pixel (true positive)
	# red   = false positive (detected edge not near any centerline)
	# blue  = false negative (centerline not matched by any edge)
	tolerance = 2
	detection = skeletonize_img(detection)
	detection = extractor(
		detection
	)  # this function ensures that channels are not detected twice (left and right)

	detection = 255 - detection
	# compute distance to ground truth centerline
	dist_to_centerline = cv2.distanceTransform(truth, cv2.DIST_L2, maskSize=0)
	dist_to_edges = cv2.distanceTransform(detection, cv2.DIST_L2, maskSize=0)

	# check which ones are within the tolerance
	true_positives = (detection == 0) & (dist_to_centerline <= tolerance)
	false_positives = (detection == 0) & (dist_to_centerline > tolerance)
	false_negatives = (dist_to_edges > tolerance) & (truth == 0)

	tp = np.sum(true_positives)
	fp = np.sum(false_positives)
	fn = np.sum(false_negatives)

	precision = tp / (tp + fp + 1e-6)
	recall = tp / (tp + fn + 1e-6)
	f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

	metrics = {'f1_score': round(f1, 6)}

	# Prepare visualization image
	if base_image is not None:
		if len(base_image.shape) == 2:
			vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
		else:
			vis = base_image.copy()
		vis = vis.astype(np.uint8)
	else:
		vis = np.ones((*detection.shape, 3), dtype=np.uint8) * 255  # white canvas

	# Apply overlay colors
	vis[true_positives] = [0, 255, 0]  # Green
	vis[false_positives] = [255, 0, 0]  # Red
	vis[false_negatives] = [0, 0, 255]  # Blue

	return vis, metrics


def kaggle_submission(detection, filename):
	row_indices = pd.DataFrame({'row': np.arange(1, detection.shape[0] + 1)})
	results = pd.DataFrame(detection)
	upload_df = pd.concat([row_indices, results], axis=1)
	upload_df.to_csv(filename, index=False)
	return upload_df


# test

