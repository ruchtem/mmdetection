from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

from pathlib import Path

config_file = 'configs/moving_mnist/mask_rcnn_simplest.py'
checkpoint_file = 'work_dirs/mask_rcnn_r50_fpn_1x/epoch_14.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file)

# test a single image and show the results

for img in Path("data/moving-mnist/val/").iterdir():

    #img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, str(img))
    # or save the visualization results to image files
    show_result(str(img), result, model.CLASSES, show=False, out_file="work_dirs/img/%s" % img.name)