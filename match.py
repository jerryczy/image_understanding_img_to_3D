import csv
import os
import sys
import time
import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.optimize import least_squares
import lm_single_step

DATA_DIR = '../dataset/Monster/data/'
# DATA_DIR = '../dataset/SculptFace/'
# DATA_DIR = '../dataset/samba_0000/'

def process_sift_data():
	'''
	Out:	K: number of image entries
			N: number of key points for a image
		name_idx: K; [(name, start index, end index)]
		features: sum_N * 4; np matrix
		discriptors: sum_N * 128; np matrix
	'''
	for subdir, dirs, files in os.walk(DATA_DIR):
		features = []
		discriptors = []
		name_idx = []
		total_lenth = 0
		for file in files[:4]:
			if file[-4:] != '.csv':
				continue
			fullFile = os.path.join(subdir, file)
			reading = []
			with open(fullFile, 'r') as fd:
				reader = csv.reader(fd)
				reading = [row for row in reader]
			print('processing {}'.format(file))
			# feature: N x 4 matrix
			feature = np.array(reading[:4]).transpose().astype(float)
			# discriptor: N x 128 matrix
			discriptor = np.array(reading[4:]).transpose().astype(float)
			# random.seed(12345)
			# random.shuffle(feature)
			# feature = feature[:2000]
			# random.shuffle(discriptor)
			# discriptor = discriptor[:2000]
			features.append(feature)
			discriptors.append(discriptor)
			name_idx.append((file[:-4], total_lenth, total_lenth + feature.shape[0]))
			total_lenth += feature.shape[0]
	return name_idx, np.concatenate(features), np.concatenate(discriptors)


def feature_matching(discriptors):
	'''
	In:
		discriptors: sum_N * 128; np matrix
	Out:
		distance: sum_N * 4 np matrix of distance, 
	entry is infinity if it is an ourlier
		indexes: sum_N * 4 np matrix of index, 
	entry is -1 if it is an ourlier
	'''
	# use K-D tree to find 4 nearst neighbors
	# use k = 6 s.t. k=0 is itself and k=6 is ourlier
	leafsize = int(len(discriptors) * 0.01) # TODO: find a good leafsize
	tree = KDTree(discriptors, leafsize=leafsize)
	print('KDTree initialized')
	start_time = time.time()
	distances, indexes = tree.query(discriptors, k=6)
	print('query completed')
	print("--- [%s] seconds ---" % (time.time() - start_time))
	# reject ourliers
	for idx, dis in enumerate(distances):
		e_outlier = dis[-1]
		for jdx, d in enumerate(dis):
			if d >= e_outlier * 0.8:
				distances[idx][jdx] = np.inf
				indexes[idx][jdx] = -1
	return distances[:, 1:-1], indexes[:, 1:-1]


# given a index, find the index of which image it belongs to
def inv_index(name_idx, idx):
	for i, data in enumerate(name_idx):
		_, start, end = data
		if idx >= start and idx < end:
			return i


# select k best matching images by number of matches
def select_matching_candidates(name_idx, indexes, m=6):
	'''
	In:
		name_idx: K; [(name, start index, end index)]
		indexes: sum_N * 4 np matrix of index
	Out:
		candinate: K * m; list of np array where each entry is a index to image
	'''
	candidate = []
	for _, start, end in name_idx:
		num_match = np.zeros(len(name_idx))
		for i in indexes[start:end]:
			for j in i:
				if j > 0:
					num_match[inv_index(name_idx, j)] += 1
		match_index = np.argpartition(num_match, len(name_idx) - m)[-m:]
		candidate.append(np.arange(len(name_idx))[match_index])
	return candidate


# calculate rows use fo the calculation of fundamental matrix
def point_pair_to_XY(pair):
	x1, y1, x2, y2 = pair
	return np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])


def svd_solver(A, b):
	u, s, vt = np.linalg.svd(A)
	w = np.linalg.solve(np.diag(s), np.dot(u.transpose(),b))
	return np.dot(vt.transpose(), w)

# run single step of RANSAC using 8 point algorithm, return number of inliers and fundamental matrix F
def ransac_single_step(in_data):
	np.random.shuffle(in_data)
	XY = np.zeros((8, 9))
	for i in range(8):
		XY[i] = point_pair_to_XY(in_data[i])
	# estimate F matrix
	f = np.linalg.svd(XY)[-1].transpose()[-1].reshape(3, 3)
	u, s, vt = np.linalg.svd(f)
	s[2] = 0
	s_new = np.diag(s)
	F = np.matmul(np.matmul(u, s_new), vt)
	# calculate inliers
	inlier = 0
	inliers = []
	for idx, pair in enumerate(in_data):
		x1, y1, x2, y2 = pair
		xy1 = np.array([x1, y1, 1])
		xy2 = np.array([x2, y2, 1])
		d = np.matmul(np.matmul(xy1.transpose(), F), xy2)
		if abs(d) < 0.1: # TODO: find a good threshold
			inlier += 1
			inliers.append(pair)
	return inlier, F, inliers, vt[-1], u.transpose()[-1]

# given image index and its corresponding k best matches, find fundamental matrix between them
def image_matching(img_idx, candidate_idx, name_idx, indexes, features, N=10000):
	'''
	In:
		img_idx: index of current image
		candidate_idx: numpy array of size m
		name_idx: K; [(name, start index, end index)]
		indexes: sum_N * 4 np matrix of index
		features: sum_N * 4; [np matrix]
		N: number of RANSAC iterations
	Out:
		fundamental_matrix: m * 3 * 3 np matrix, each row is a flat fundamental matrix
	'''
	img_name, img_start, img_end = name_idx[img_idx]
	fundamental_matrix = []
	inlier_number = []
	inlier_set = []
	epipolar_set = []
	for c in candidate_idx:
		# for each pair of matched images,
		# calculate fundamental matrix
		cand_name, cand_start, cand_end = name_idx[c]
		XY = []
		for i in range(img_start, img_end):
			# for each key point in current image, 
			# find its inlier matched points in this candidate image
			x1, y1, _, _ = features[i]
			for j in indexes[i]:
				if j < cand_end and j >= cand_start:
					x2, y2, _, _ = features[j]
					XY.append([x1, y1, x2, y2])
		print('XY: {}'.format(len(XY)))
		# Use RANSAC
		# TODO estimate number of iteration needed
		# inlier, _ = ransac_single_step(XY)
		# print([inlier, len(XY)])
		# num_iteration = int(np.log(0.05)/np.log(1-(inlier/len(XY))**len(XY)))
		# print(num_iteration)
		# use RANSAC to approximate F
		best_val = 0
		best_F = np.zeros((3, 3))
		best_inliers = []
		best_e = []
		for _ in range(N):
			inlier, F, inliers, e1, e2 = ransac_single_step(XY)
			if inlier > best_val:
				best_val = inlier
				best_F = F
				best_inliers = inliers
				best_e = [e1, e2]
		fundamental_matrix.append(best_F)
		inlier_number.append(best_val)
		inlier_set.append(best_inliers)
		epipolar_set.append(best_e)
		print('inliers: {}'.format(best_val))

	k = np.argmax(inlier_number)

	# plot epipolar line
	print(epipolar_set[k][0]/epipolar_set[k][0][-1])
	print(epipolar_set[k][1]/epipolar_set[k][1][-1])
	e1x, e1y, _ = epipolar_set[k][0]/epipolar_set[k][0][-1]
	e2x, e2y, _ = epipolar_set[k][1]/epipolar_set[k][1][-1]
	img1 = cv2.imread(os.path.join(DATA_DIR, img_name + '.jpg'))
	img2 = cv2.imread(os.path.join(DATA_DIR, name_idx[k][0] + '.jpg'))
	for pair in inlier_set[k]:
		x1, y1, x2, y2 = pair
		cv2.circle(img1, (int(x1), int(y1)), 1, (255, 255, 255), 5)
		cv2.circle(img2, (int(x2), int(y2)), 1, (255, 255, 255), 5)
		rect1 = (0, 0, img1.shape[1], img1.shape[0])
		_, pt1, pt2 = cv2.clipLine(rect1, (int(x1), int(y1)), (int(e1x), int(e1y)))
		cv2.line(img1, pt1, pt2, (255, 255, 255), 2)
		rect2 = (0, 0, img2.shape[1], img2.shape[0])
		_, pt1, pt2 = cv2.clipLine(rect2, (int(x2), int(y2)), (int(e2x), int(e2y)))
		cv2.line(img2, pt1, pt2, (255, 255, 255), 2)
	cv2.imwrite(img_name + '_' + img_name + '_1.jpg', img1)
	cv2.imwrite(img_name + '_' + name_idx[k][0] + '_1.jpg', img2)

	# sort fundamental matrix by number of inliers
	# n = len(inlier_number)
	# lst = list(range(n))
	# lst.sort(key=inlier_number.__getitem__)
	# fundamental_matrix = list(map(fundamental_matrix.__getitem__, lst))
	# inlier_number = list(map(inlier_number.__getitem__, lst))
	# inlier_set = list(map(inlier_set.__getitem__, lst))
	return candidate_idx[k], fundamental_matrix[k], inlier_number[k], inlier_set[k]

	# return fundamental_matrix, inlier_number, inlier_set


def component_track(image_pair):
	# find a linked series of images that match one to the other
	linked = [0]
	new = image_pair[0][0]
	while new not in linked:
		linked.append(new)
		new = image_pair[new][0]
	print(linked)
	# find components track
	# initialized as first image-candidate match
	for x1, y1, x2, y2 in image_pair[0][-1]:
		components.append([(x1, y1), (x2, y2)])
	for idx in linked[1:]:
		inlier_set = image_pair[idx][-1]
		for x1, y1, x2, y2 in inlier_set:
			new_entry = True
			for comp in components:
				if (x1, y1) in comp and (x1, y1) == comp[-1]:
					comp.append((x2, y2))
					new_entry = False
				else:
					comp.append(None)
			if new_entry:
				lst = [None for _ in range(idx-1)] # fill missing track
				lst.extend([(x1, y1), (x2, y2)])
				components.append(lst)
	return components, linked


def partition(lst):
	start = 0
	end = len(lst)
	for idx, p in enumerate(lst):
		if p:
			start = idx
			break
	for idx, p in enumerate(reversed(lst)):
		if p:
			end = -idx
			break
	if end == 0:
		end = len(lst)
	lst.reverse()
	return start, end


def bundle_adjustment(lst, linked):
	start, end = partition(lst)
	lst = lst[start:end]
	linked = linked[start:end]
	def residual(x):
		a, b, c = x

		return np.array([0, 0, 0])


	least_squares(residual, x0, method='lm', loss='huber')


if __name__ == '__main__':
	start_time = time.time()
	name_idx, features, discriptors = process_sift_data()
	print(features.shape)
	print(discriptors.shape)
	print("--- %s seconds ---" % (time.time() - start_time))
	distances, indexes = feature_matching(discriptors)
	print("--- %s seconds ---" % (time.time() - start_time))
	sys.stdout.flush()
	candidates = select_matching_candidates(name_idx, indexes, m=1)
	image_pair = []
	for i, data in enumerate(name_idx):
		img_name, start, end = data
		print('finding match for {}'.format(img_name))
		matched_img, fundamental_matrix, inlier_number, inlier_set = \
			image_matching(i, candidates[i], name_idx, indexes, features)
		image_pair.append((matched_img, fundamental_matrix, inlier_number, inlier_set))
		print("--- %s seconds ---" % (time.time() - start_time))
	components, linked = component_track(image_pair)
	print("--- %s seconds ---" % (time.time() - start_time))
	for comp in component:
		bundle_adjustment(comp, linked)
