# Face_Pytorch

- 참고: https://github.com/wujiyang/Face_Pytorch

---

## 1. dataset
- row 이미지 사용
#### trainset
- train.rec를 이미지로 변환하여 imgs폴더에 저장한 후, 이를 train에 활용함
- 이미지 dir로 만든 .lst파일을 사용함.(download한 .lst파일은 맞지 않음)
- dataset: faces_wecface_112x112
- dowonload: https://www.dropbox.com/s/lfluom5ybqqln02/faces_CASIA_112x112.zip?dl=0
- 장점: 이미지 자체로 학습하므로 이미지에 대한 전처리(.lst, .rec, .idx 파일)을 만들 필요가 없고, 직관적임
- 단점: train, test set을 직접 만들거나 관리할 때 용량이 크고 번거로움

#### testset
- aligned lfw, agedb, cfpfp set (112x112)
- 각 testset별로 pairs.txt 파일 필요

## 2.train

- dataset size:  490623 / 10572