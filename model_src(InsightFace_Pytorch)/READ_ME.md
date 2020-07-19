# InsightFace-pytorch

- https://github.com/TreB1eN/InsightFace_Pytorch

---

## 1. dataset
- download: https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0
- prepare_data.py이용
#### trainset
- mxnet에서 만들어진 .rec, .lst, .idx 파일을 img로 변환
- dir: 85742, file: 5822653
```
faces_emore/imgs/num (num: 0~85742)
		   /train.rec
		   /train.idx
		   /property

```
#### testset
- agedb_30 :  12000
- cfp_fp : 14000
- lfw : 12000
- calfw : 12000
- cfp_ff :  14000
- cplfw : 12000
- vgg2_fp: 10000

```
# agedb dataset
faces_emore/agebd_30/meta/sizes
						 /storage
					/data/__num.blp ( num: 0~923)
           /agedb_30_list.npy
		   /agedb_30.bin					
                    
```

   