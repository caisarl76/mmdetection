## 1. 데이터셋 생성

./mmdetection/mmdet/datasets/{DATASET_NAME.py} 파일 생성 (pantos_ver2.py 복사 권장)

class 이름 변경, class내  METAINFO 정보 변경 ( 클래스 이름, 개수 확인)

./mmdetection/mmdet/datasets/__init__.py 데이터셋 클래스 등록
import 코드 추가
__all__ 리스트에 데이터셋 클래스 이름 추가


## 2. 학습 모델 config 구성
./mmdetection/custom_configs/final/config.py 생성 (90_R101_2x_1.py 복사 권장)

아래 값 1. 에서 선언한 사용할 데이터셋에 맞게 변경
line 4~6" train_ann, val_ann, img_path 
line 25: dataset_type
line 26: data_root  


## 3. 모델 학습 
./mmdetection/custom_configs/final/config.py 
line 7: train_batch_size -> 학습 환경에 맞게 수정

터미널 상에서 학습 코드 실행:
```python tools/train.py custom_configs/final/config.py```

정상적으로 학습이 진행된다면 터미널 상에 아래와 같은 로그가 표시될 것입니다.

```
2023/11/29 02:18:39 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
2023/11/29 02:18:39 - mmengine - INFO - Checkpoints will be saved to /root/mmdetection/work_dirs/90_R101_2x_1.
2023/11/29 02:19:30 - mmengine - INFO - Epoch(train)  [1][ 50/663]  lr: 1.9820e-03  eta: 4:27:58  time: 1.0137  data_time: 0.0554  memory: 20138  loss: 0.7690  loss_rpn_cls: 0.3936  loss_rpn_bbox: 0.0343  loss_cls: 0.3214  acc: 98.5840  loss_bbox: 0.0197
2023/11/29 02:20:20 - mmengine - INFO - Epoch(train)  [1][100/663]  lr: 3.9840e-03  eta: 4:25:17  time: 0.9996  data_time: 0.0519  memory: 20138  loss: 0.4335  loss_rpn_cls: 0.0921  loss_rpn_bbox: 0.0311  loss_cls: 0.1786  acc: 96.2646  loss_bbox: 0.1318
```

## 4. 학습 결과 확인 및 파일 access
default로 지정된 저장경로: ./mmdetection/work_dirs/{CONFIG_NAME}/ 에 (예: ./mmdetection/work_dirs/90_R101_2x_1/) 모델 weight (epoch_*.pth) 와 학습 log 확인할 수 있습니다.

이후 해당 model weight로 inference 과정에 경로를 입력해주시면 됩니다. 
