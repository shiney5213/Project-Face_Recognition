# Project-Face_Recognition
 딥러닝기술을 활용하여 사람들의 얼굴을 인식합니다.

 InshgitFace_Pytorch를 korean_face_dataset을 추가하여 train합니다

---

## 1. dataset

#### 1.1. training set

##### 1.1.1. face_emore

- download: Dataset Zoo(https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
- MS1M_ArcFAce:  dir: 85742, file: 5822653

##### 1.1.2. k_face_dataset

  - download: AI Hub(http://www.aihub.or.kr/)
  - 한 사람당 image가 32,400장으로 너무 많아 조명(L1, L2, L3), 액세서리(S001, S002, S005), 방향(C5~C10), 표정(E1, E2, E3) 데이터 사용
  - dir: 400, file: 49,491, (132 * 400 = 52800 중 나머지 image는 face deteching 실패)
  - dir: 19062421 ~19101513

##### 1.1.3. all data

  - dir: 86142, file: 5872144
- preprocessing
  - face_emore: .rec, .idx -> imgs dir ([source](https://github.com/shiney5213/Project-Face_Recognition/blob/master/kface_data_src/1.6.rec2img.py))
  - k_face_dataset: aligned_112x112 by MTCNN([source](https://github.com/shiney5213/Project-Face_Recognition/blob/master/kface_data_src/1.2.face_crop_112x112.py))

#### 1.2. test set

##### 1.2.1.  face_emore

  - agedb_30 :  12000
  - cfp_fp : 14000
  - calfw : 12000
  - cfp_ff :  14000
  - cplfw : 12000
  - vgg2_fp: 10000
  - lfw : 12000

##### 1.2.1.  k_face_dataset

  - dir: 400, file: 12000 ( 400 * 32 = 12800 개 중 800장은 face deteching 실패)

---
## 2. model
- https://github.com/TreB1eN/InsightFace_Pytorch
#### 1. backborn
- IR-SE50,  Mobilefacenet 중 택1
- epoth 