"this is the file to be called in order do embed"
"just call embedding"

from scipy.fft import dct, idct
import numpy as np
import hvs_lambda as hvs
import embedding_flat_file as fl
from psnr import similarity, wpsnr
import cv2
import pywt

# Functions used in the embedding

def im_dct(image):
	return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')


def im_idct(image):
	return (idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho'))


def embedding_SVD(image, mark, alpha, mode='additive'):
	"""
	Embedding of the mark in the singular values of image
	:param image: image as nxn numpy matrix
	:param mark: mark as numpy array, of dimension < n
	:param alpha: strength of the embedding
	:param mode: multiplicative, additive and the third option is an adaptive version of the additive
	:return: watermarked image as nxn numpy matrix
	"""
	# diagonalization
	u, s, v = np.linalg.svd(image)
	if mark.size >= s.size:
		print('error mark', mark.size, 'diag', s.size)
		return 123
	if mode == 'multiplicative':
		mark_pad = np.pad(mark, (1, s.size - mark.size - 1), 'constant')
		s *= 1 + alpha * mark_pad
	elif mode == 'additive':
		mark_pad = np.pad(mark, (1, s.size - mark.size - 1), 'constant')
		s += alpha * mark_pad
	else:
		# select the strongest singular value and modify them (except the greatest one)
		locations = np.argsort(-s)
		for i in range(len(mark)):
			s[locations[i + 1]] += alpha * mark[i]

	watermarked = np.matrix(u) * np.diag(s) * np.matrix(v)
	return watermarked





def embedding(name_image, name_mark, alpha=10, name_output='watermarked.bmp', dim=8, step=20, max_splits=500, min_splits=170,
			  sub_size=6, Xi_exp = 0.2, Lambda_exp = 0.3, L_exp = 0.2, ceil = True):
	"""
	Adaptive embedding function based on HVS (Human Visual System) to distinguish the good blocks for the embedding.
	Then it embeds the mark in dim x dim block in the LL level of DWT by doing an SVD embedding on the DCT
	the intensity of the embedding depends on the characteristics of the block.
	If the image is too flat we select the low luminance flat areas and embed on the first 3 AC coefficients one bit of the mark
	:param name_image: path to image
	:param mark: mark as sequence of +1 and -1
	:param alpha: strength of embedding
	:param name_output: path for the output
	:param dim: dimension of blocks in first level DWT
	:param step: quantization steps for hws function
	:param max_splits: max number of blocks to be embedded
	:param min_splits: min number of blocks to be embedded
	:param sub_size: max dimension of submarks
	:param Xi_exp: represent how much we weight Xi from HVS
	:param Lambda_exp: represent how much we weight Lambda from HVS
	:param L_exp: represent how much we weight L from HVS
	:param ceil: if we use ceil or rint when quantizing the HVS
	:return: the embedded images as numpy matrix
	"""
	MARK=np.load(name_mark)
	mark=np.array([(-1)**m for m in MARK])
	
	image = cv2.imread(name_image, 0)
	# evaluate parameters of Human visual system
	q = hvs.hvs_step(image, dim=dim, step=step, Xi_exp = Xi_exp, Lambda_exp = Lambda_exp, L_exp = L_exp, ceil = ceil)

	# first level DWT
	image, (LH, HL, HH) = pywt.dwt2(image, 'haar')
	sh = image.shape

	if sh[0] % dim != 0:
		return 'img size not div by ' + str(dim)


	splits = min(np.count_nonzero(q), max_splits)

	# case of flat images
	# we short the mark and proceed with a different embedding strategy for the second part
	if splits < min_splits:
		new_mark_size = int(splits * sub_size - 1)
		flat_mark_size = mark.size - new_mark_size
		mark_flat = mark[new_mark_size:]
		mark = mark[:new_mark_size]
		dc_coeff = q.copy()
		dc_coeff[:] = 0
		# creation of the coefficients to decide low luminance flat blocks
		for i in range(dc_coeff.shape[0]):
			for j in range(dc_coeff.shape[1]):
				if q[i, j] != 0:
					dc_coeff[i, j] = 99999
				else:
					dct = im_dct(image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim])
					dc_coeff[i, j] = dct[0, 0] ** 0.2 * np.var(np.squeeze(dct)[1:])

		dark_locations = np.argsort(dc_coeff, axis=None)
		dark_locations = dark_locations[:flat_mark_size]
		rows = dc_coeff.shape[0]
		dark_locations = [(val // rows, val % rows) for val in dark_locations]
		mark_pos = 0

		# embedding in the dark flat locations using a modified version of DCT Spread spectrum
		for loc in dark_locations:
			i = loc[0]
			j = loc[1]
			image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = fl.embedding_flat(
				image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim],
				wat=mark_flat[mark_pos])
			mark_pos += 1

	# selection of the best locations for embedding
	locations = np.argsort(-q, axis=None)
	locations = locations[:splits]
	rows = q.shape[0]
	locations = [(val // rows, val % rows) for val in locations]

	# splitting of the mark and padding of the shorter submarks
	sub_mark = np.array_split(mark, splits)
	# print('num of submarks', len(sub_mark))
	sub_mark_size = sub_mark[0].size
	for i in range(len(sub_mark)):
		sub_mark[i] = np.pad(sub_mark[i], (0, sub_mark_size - sub_mark[i].size), 'constant')

	# embedding of the mark in the most adapt locations
	for loc in locations:
		i = loc[0]
		j = loc[1]
		image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = im_idct(
			embedding_SVD(
				im_dct(image[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim]),
						  sub_mark.pop(0), alpha=(q[i, j]) * alpha, mode="d"))

	# print('em splits', splits, '| submarksize', sub_mark_size, '| flat size',flat_mark_size)

	watermarked = pywt.idwt2((image, (LH, HL, HH)), 'haar')
	# write of the output images
	cv2.imwrite(name_output, watermarked)
	return watermarked

