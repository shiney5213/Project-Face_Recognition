from mtcnn import MTCNN
import cv2
import os 
import matplotlib.pyplot as plt

def face_detection(img):
    
    img_height, img_width = img.shape[:2]
#     print(img_height, img_width)
    detector = MTCNN()
    results = detector.detect_faces(img)
    
    if len(results) >= 1:
#         print('len(results)', len(results))
        
        if len(results)== 1:
            faces = results[0]['box']
            # x, y, width, height
            x, y, w, h = faces
#             print(x, y, w, h)
            
            
            left = x if x >0 else 0
            top = y if y>0 else 0
            right = x + w
            bottom = y + h
            x_center= left + int(w/2)
            left = max(0, x_center - int(h/2))
            right = min( x_center + int( h/2), img_width)
#             print(left, right, top, bottom)
            img_square = img[top:bottom, left:right]
#             print(img_square.shape)
            img_112 = cv2.resize(img_square, dsize=(112,112), interpolation = cv2.INTER_AREA)
#             imshow('square', img_112) 
            return True, img_112      
            
        
        else:
            face_area = []
            for i, result in enumerate(results):
                faces = result['box']
                x, y, w, h = faces
                face_area.append((i, w*h))
            
            print(face_area)
            area_sorted = sorted(face_area, key = lambda x: x[1], reverse=True)
            print(area_sorted)
            
            
            max_index = area_sorted[0][0]
#             print(max_index)
            max_faces = results[max_index]['box']
            x, y, w, h = max_faces
#             print(x, y, w, h)
            
            left = x if x >0 else 0
            top = y if y>0 else 0
            right = x + w
            bottom = y + h
            x_center= left + int(w/2)
            left = max(0, x_center - int(h/2))
            right = min( x_center + int( h/2), img_width)
#             print(left, right, top, bottom)
            img_square = img[top:bottom, left:right]
#             print(img_square.shape)
            img_112 = cv2.resize(img_square, dsize=(112,112), interpolation = cv2.INTER_AREA)
#             imshow('square', img_112) 
            return True, img_112             
    else:
        return False, None

def main():
	old_dir_path = '../data/my_kface/test'
	new_dir_path = '../data/my_kface/kface_test_112x112'
	# old_dir_path = '../data/raw/vgg2_test'
	# new_dir_path = '../data/raq/vgg2_test_align'
	
	old_dir_list = os.listdir(old_dir_path)

	for i , people_name in enumerate(old_dir_list):
#     print(old_dir)
		new_people_dir = os.path.join(new_dir_path, people_name)
	#     print(new_people_dir)
		try:
			os.mkdir(new_people_dir)
		except Exception as err:
			pass
        
		people_path = os.path.join(old_dir_path, people_name)
		old_file_list = os.listdir(people_path)
		not_find = 0
		find_face = 0
		for j, file in enumerate(old_file_list):
			old_file_path = os.path.join(people_path, file)
			print(i, 'old',old_file_path)
			
			original_image = cv2.imread(old_file_path, 1)
			ret, img_112 = face_detection(original_image)
			
			if ret:
				new_file_path = f"{new_people_dir}/{file}"
				print('new', new_file_path)
				cv2.imwrite(new_file_path, img_112)
				find_face += 1
			else:
				not_find +=1
		print(f'{people_name}-find:{find_face},not_find:{not_find}' )


if __name__ == '__main__':
	main()


