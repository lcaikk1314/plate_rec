import sys, os
import keras
import cv2

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':
	
	input_dir  = sys.argv[1]
	output_dir = input_dir

	lp_threshold = .5

	wpod_net_path = sys.argv[2]
	wpod_net = load_model(wpod_net_path)##########需要看

	imgs_paths = glob('%s/*car.png' % input_dir)
	print('%s/*car.png' % input_dir)
	print( 'Searching for license plates using WPOD-NET')
	print( imgs_paths)
	for i,img_path in enumerate(imgs_paths):

		print( '\t Processing %s' % img_path)

		bname = splitext(basename(img_path))[0]
		Ivehicle = cv2.imread(img_path)

		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)#此行有问题，如果是为了取整，应该是减法‘-’，小于608且是16的整数倍
		print( "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

		Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)########需要看

		if len(LlpImgs):
			Ilp = LlpImgs[0]
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

			s = Shape(Llp[0].pts)

			cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
			writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

