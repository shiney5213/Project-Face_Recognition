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

  - dir: 400, file: 11027 ( 400 * 32 = 12800 개 중 1773장은 face deteching 실패)

---
## 2. model
- https://github.com/TreB1eN/InsightFace_Pytorch
#### 1. backborn
- IR-SE50,  Mobilefacenet 중 택1
- epoth 




---
## Question?!
#### 1. 보통 한 사람의 img를 train, test으로 나누어 학습, 예측에 사용하는데 Ms1m dataset에는 lwf, ageDB 등 다양한 데이터셋의 사람들이 포함되어있을까?
- 코드를 살펴보면 Ms1m dataset으로 학습을 진행 한 후에 lfw등 testset을 불러와서 모델을 통과한 embedding값의 거리를 측정하여 test를 하고 있음.
- 이 경우에는 train한 img가 아니더라도 test가 가능함 => One Shot Learning이기 때문
#### 2. Korean face dataset을 학습시킬 때는 어떻게 하는 것이 좋을까?

- 일단 학습한 ms1m 데이터셋은 서양인 위주이기 때문에 train data에 동양인(asian, korean)의 얼굴이 포함되어야 할 것 같아, Asian-Celeb dataset이나 korean face dataset을 포함하여 train을 진행할 예정임.

- test set에는 trainset에 포함되지 않은 korean dataset으로 예측할 예정임.

#### 3. Validation set이 없은 걸까?
- 현재 코드에는 validation set을 사용하는 부분이 보이지 않음. 
- 엄연히 말하면 test set이 validation set의 역할을 한다고도 볼 수 있음.
- 이 부분은 좀 더 알아봐야겠음.

#### 4. test를 할 때에도 arcface header를 사용할까?
- 코드에서는 trained model만 사용함
- 왜 test할 때는 header를 사용하지 않는지 알아봐야겠음.

#### 5. testset에서 찾는 best threshold값이 dataset마다 다른데 이를 어떻게 활용할까?

#### 6. pytorch에서는 model 뿐 아니라 optimizer를 왜 저장할까?
- optimizer의 매개변수도 반복 후에 변경될 수 있으므로 함께 저장해야 함.

#### 7. scheduler도 저장해야 할까?

#### 8. validation loss는 구하지 않아도 되는걸까?
- overfitting 여부때문에 알아야 할 것 같음