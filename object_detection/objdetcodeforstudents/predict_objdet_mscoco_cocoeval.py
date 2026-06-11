import torchvision.models.detection
import torch
import json

#wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
#wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

def run():

  debugmaxnum = -1

  #https://pytorch.org/vision/stable/models.html  
  
  #modelweights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
  #model =   torchvision.models.detection.retinanet_resnet50_fpn_v2(weights = modelweights)

  modelweights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=modelweights, box_score_thresh=0.9)

  #modelweights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
  #model =   torchvision.models.detection.ssd300_vgg16(weights = modelweights)
  
  transforms = modelweights.transforms()
  
  #cocodspath = '/media/binder/6cd6f955-900d-4f12-9915-07507e4ea5f7/data2023/coco/val2017/'
  #annotfile = '/media/binder/6cd6f955-900d-4f12-9915-07507e4ea5f7/data2023/coco/annotations/instances_val2017.json'
  cocodspath = '/home/binder/experiments/datasets/coco/val2017'
  annotfile = '/home/binder/experiments/datasets/coco/annotations/instances_val2017.json'
  ds_val  = torchvision.datasets.CocoDetection(root= cocodspath, annFile = annotfile, transform = transforms) 
 
  ds_val_loader=  torch.utils.data.DataLoader(ds_val, batch_size= 1, shuffle=False )

  cocoAnnotation = get_coco_api_from_dataset(ds_val_loader.dataset)
  coco_evaluator2 = CocoEvaluator( cocoAnnotation , iou_types = ['bbox'])

  print('done creating cocoAnnotation and coco_evaluator2')
   
  model.eval()
  model.to('cuda:0') #outcomment if needed

  n_threads = torch.get_num_threads()

  with torch.no_grad():
  
    for i, (img,target) in enumerate(ds_val_loader): 
   
      #print(target[0]) #  'area': tensor([531.8071], dtype=torch.float64), 'iscrowd': tensor([0]), 'image_id': tensor([139]), 'bbox': [tensor([236.9800], dtype=torch.float64), tensor([142.5100], dtype=torch.float64), tensor([24.7000], dtype=torch.float64), tensor([69.5000], dtype=torch.float64)], 'category_id': tensor([64]), 'id': tensor([26547])}

      if (debugmaxnum > 0) and (i >= debugmaxnum): 
        break
      predsth = False
              
      img = img.to('cuda:0')   #outcomment if needed                                      
      predictions = model(img)
      predicted = predictions[0] # unbatching

      print(len(predictions))
      print(predicted)
      #exit()

      print('num gt boxes, num pred boxes', len(target), predicted['boxes'].shape[0])

      if (predicted['boxes'].shape[0]>0) and len(target)>0:  
        
        #TODO        
        predsdic = { 
        'labels': predicted['labels'].to('cpu'), 
        'boxes': predicted['boxes'].to('cpu'),
        'scores': predicted['scores'].to('cpu'),        
         }
         
        res = { target[0]["image_id"].item(): predsdic}
        coco_evaluator2.update(res)
        predsth = True

    # print AP
    if predsth:
      coco_evaluator2.accumulate()
      listofit=coco_evaluator2.summarize()   
      #print('listofit',listofit)             
      #accuracy = listofit[0]


if __name__=='__main__':
  run()  


    
