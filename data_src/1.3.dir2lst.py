import sys
import os
from easydict import EasyDict as edict

def main():
	# input_dir ='../data/small_vgg/small_vgg_112x112/train'
	input_dir ='../data/my_vgg/aligned_vgg'
	lst_save = '../data/my_vgg/aligned_vgg.lst'


	ret = []
	label = 0
	person_names = []
	for person_name in os.listdir(input_dir):
		person_names.append(person_name)
	person_names = sorted(person_names)

	for person_name in person_names:
		# persondir_path
		_subdir = os.path.join(input_dir, person_name)
		print(_subdir)
		if not os.path.isdir(_subdir):
			continue
		
		_ret = []
		for img in os.listdir(_subdir):
			# imgfile_path
			fimage = edict()
			fimage.id = os.path.join(_subdir, img)
			fimage.classname = str(label)
			fimage.image_path = os.path.join(_subdir, img)
			fimage.bbox = None
			fimage.landmark = None
			_ret.append(fimage)
		ret += _ret
		label+=1

	with open(lst_save, 'a', encoding = 'utf-8') as f:
		for item in ret:
			item.image_path = item.image_path.replace('\\', '/')
			f.write("%d\t%s\t%d\n" % (1, item.image_path, int(item.classname)))

	


if __name__ == '__main__':
	main()
