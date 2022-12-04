import cv2 
import numpy as np 

# Load image 
img = cv2.imread('image.jpg') 

# Get size of image 
h, w, c = img.shape 

# Get uniform-LBP of image 
ulbp = np.zeros((h, w), dtype=np.uint8) 

for i in range(1, h - 1): 
	for j in range(1, w - 1): 
		ulbp[i, j] = ((img[i - 1, j - 1] > img[i, j]) << 0) + ((img[i - 1, j] > img[i, j]) << 1) + ((img[i - 1, j + 1] > img[i, j]) << 2) + ((img[i, j + 1] > img[i, j]) << 3) + ((img[i + 1, j + 1] > img[i, j]) << 4) + ((img[i + 1, j] > img[i, j]) << 5) + ((img[i + 1, j - 1] > img[i, j]) << 6) + ((img[i, j - 1] > img[i, j]) << 7) 

# Calculate histogram 
hist = np.zeros(256, dtype=np.float32) 
for i in range(h): 
	for j in range(w): 
		hist[ulbp[i, j]] += 1 

# Normalize histogram between [0, 1] 
hist /= (h * w) 

# Calculate similarity by Manhattan distance 
def manhattan_distance(h1, h2): 
	return np.sum(np.abs(h1 - h2)) 

# Load other images
imgs = [cv2.imread('image1.jpg'), cv2.imread('image2.jpg'), cv2.imread('image3.jpg')] 

# Calculate similarity 
scores = [] 
for img_ in imgs: 
	# Get uniform-LBP of image 
	ulbp_ = np.zeros((h, w), dtype=np.uint8) 
	for i in range(1, h - 1): 
		for j in range(1, w - 1): 
			ulbp_[i, j] = ((img_[i - 1, j - 1] > img_[i, j]) << 0) + ((img_[i - 1, j] > img_[i, j]) << 1) + ((img_[i - 1, j + 1] > img_[i, j]) << 2) + ((img_[i, j + 1] > img_[i, j]) << 3) + ((img_[i + 1, j + 1] > img_[i, j]) << 4) + ((img_[i + 1, j] > img_[i, j]) << 5) + ((img_[i + 1, j - 1] > img_[i, j]) << 6) + ((img_[i, j - 1] > img_[i, j]) << 7) 

	# Calculate histogram 
	hist_ = np.zeros(256, dtype=np.float32) 
	for i in range(h): 
		for j in range(w): 
			hist_[ulbp_[i, j]] += 1 

	# Normalize histogram between [0, 1] 
	hist_ /= (h * w) 

	# Calculate similarity 
	score = manhattan_distance(hist, hist_) 
	scores.append(score) 

# Find most similar images 
indexes = np.argsort(scores)[:3] 

# Print result 
for i in indexes: 
	print('Image {} is the most similar.'.format(i))