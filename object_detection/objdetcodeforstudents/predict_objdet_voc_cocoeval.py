import copy

import PIL.Image
from torchvision.utils import draw_bounding_boxes

from torchvision.transforms.functional import to_tensor
import torchvision.models.detection
import torch

from torchvision import tv_tensors

#wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
#wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

class torchvisionvoc_coco_bridge(torch.utils.data.Dataset):

  def __init__(self, vocdet_dataset, transforms= None):
    super().__init__()

    self.ds = vocdet_dataset
    self.transforms = transforms

    # needs for every image:
    #
    #   'image_id', given by idx
    #   'iscrowd' False for every box
    # and non-trivial data:
    #   'labels' for every box
    #   'area' for every box
    #   'boxes' for every box

    #https://github.com/amikelive/coco-labels
    
    self.cocolabeldict={0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'street sign',
 13: u'stop sign',
 14: u'parking meter',
 15: u'bench',
 16: u'bird',
 17: u'cat',
 18: u'dog',
 19: u'horse',
 20: u'sheep',
 21: u'cow',
 22: u'elephant',
 23: u'bear',
 24: u'zebra',
 25: u'giraffe',
 26: u'hat',
 27: u'backpack',
 28: u'umbrella',
 29: u'shoe',
 30: u'eye glasses',
 31: u'handbag',
 32: u'tie',
 33: u'suitcase',
 34: u'frisbee',
 35: u'skis',
 36: u'snowboard',
 37: u'sports ball',
 38: u'kite',
 39: u'baseball bat',
 40: u'baseball glove',
 41: u'skateboard',
 42: u'surfboard',
 43: u'tennis racket',
 44: u'bottle',
 45: u'plate',
 46: u'wine glass',
 47: u'cup',
 48: u'fork',
 49: u'knife',
 50: u'spoon',
 51: u'bowl',
 52: u'banana',
 53: u'apple',
 54: u'sandwich',
 55: u'orange',
 56: u'broccoli',
 57: u'carrot',
 58: u'hot dog',
 59: u'pizza',
 60: u'donut',
 61: u'cake',
 62: u'chair',
 63: u'couch',
 64: u'potted plant',
 65: u'bed',
 66: u'mirror',
 67: u'dining table',
 68: u'window',
 69: u'desk',
 70: u'toilet',
 71: u'door',
 72: u'tv',
 73: u'laptop',
 74: u'mouse',
 75: u'remote',
 76: u'keyboard',
 77: u'cell phone',
 78: u'microwave',
 79: u'oven',
 80: u'toaster',
 81: u'sink',
 82: u'refrigerator',
 83: u'blender',
 84: u'book',
 85: u'clock',
 86: u'vase',
 87: u'scissors',
 88: u'teddy bear',
 89: u'hair drier',
 90: u'toothbrush',
 91: u'hair brush',
 }

    # for remapping of coco classes to background or voc classes
    self.unsorted_classlist=['__background__'] #all label names
    self.unsorted_classlist2labelindex = {'__background__':0}

    self.bboxcoords=[]
    self.clslabels=[]
    self.origareas=[]
    

    for i in range(len(self.ds)):
      annotdic = self.ds[i][1]['annotation']
      filename = annotdic['filename']    

      objects = annotdic['object'] # list of #{'name': 'chair', 'pose': 'Rear', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '263', 'ymin': '211', 'xmax': '324', 'ymax': '339'}}

      curlb=[]
      curareas =[]
      curbboxes=[]

      for boxinstance in objects: # the boundingboxes in one image
        #print(boxinstance) # a dictionary like {'name': 'person', 'pose': 'Frontal', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '148', 'ymin': '108', 'xmax': '213', 'ymax': '187'}}
        
        bbox = [ int(boxinstance['bndbox']['xmin']) , int(boxinstance['bndbox']['ymin']), int(boxinstance['bndbox']['xmax']) , int(boxinstance['bndbox']['ymax'])   ]
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

        if area > 0:
          lbname = boxinstance['name'] 
          if lbname not in self.unsorted_classlist:
            self.unsorted_classlist.append(lbname)
            self.unsorted_classlist2labelindex[lbname] = len(self.unsorted_classlist)

          label_as_integer = self.unsorted_classlist2labelindex[lbname]
          
          curlb.append(label_as_integer)
          curareas.append(area)
          curbboxes.append(bbox)

      if len(curlb)>0:
        self.clslabels.append( torch.as_tensor(curlb, dtype= torch.int16) )
        
        # needs to be h,w
        imgsize  = ( int(annotdic['size']['height']), int(annotdic['size']['width']))
        wrappedboxes = torchvision.tv_tensors.BoundingBoxes( torch.tensor(curbboxes) , format="XYXY", canvas_size= imgsize )
        self.bboxcoords.append( wrappedboxes )
        self.origareas.append(  torch.tensor( curareas ) )
      else:
        self.clslabels.append( torch.tensor([]) )
        self.bboxcoords.append( torch.tensor([]) )
        self.origareas.append( torch.tensor([]) )

    #print(self.unsorted_classlist)
    #exit()
    #['background', 'chair', 'car', 'horse', 'person', 'bicycle', 'cat', 'dog', 'train', 'tvmonitor', 'bird', 'bottle', 'motorbike', 'pottedplant', 'sofa', 'sheep', 'cow', 'aeroplane', 'boat', 'bus', 'diningtable']

    num_cococls = len(self.cocolabeldict.keys())
    self.cocointlabels2vocintlabels=[0 for i in range(num_cococls)]

    remapdict={
    'couch': 'sofa',
    'tv': 'tvmonitor',
    'motorcycle':'motorbike',
    'dining table':'diningtable',
    'potted plant': 'pottedplant',
     'airplane' :'aeroplane',    
    }

    numclasses_invoclabels = 0
    includednames=set()
    for i in range(num_cococls):
      #print('mapping test index', i)
      #print( i in self.cocolabeldict.keys() )
      if i in self.cocolabeldict.keys():
      
        name = self.cocolabeldict[i]
        #        print(name,' in voc classlist ? ', name in self.unsorted_classlist )
        if name in remapdict.keys():
        
          remapped_name = remapdict[name]
          self.cocointlabels2vocintlabels[i]= self.unsorted_classlist2labelindex[remapped_name]

          numclasses_invoclabels+=1          

        elif name in self.unsorted_classlist:
          
          self.cocointlabels2vocintlabels[i]= self.unsorted_classlist2labelindex[name]
          
          numclasses_invoclabels+=1
          includednames.add(name)
        else:
          pass
          #remap to voc background, see init above, already done implicitly 
           
    print('numclasses_invoclabels',numclasses_invoclabels)
    print('voc classes not in coco: (these need the remapping)', set(self.unsorted_classlist)-includednames  )
    
    #'couch'->'sofa'
    #'tv'-> 'tvmonitor'
    #'motorcycle'->'motorbike'
    #'dining table'->'diningtable'
    #'potted plant'-> 'pottedplant'
    # 'airplane' -> 'aeroplane'
    #
    #exit()
  def remaptensor_coco2voclabels_cpudevice(self,cocolabels):
    
    #fucking ugly hack :)
    cocolabels.apply_(lambda i: self.cocointlabels2vocintlabels[i])
      
    return cocolabels

  def __len__(self):
    return len(self.clslabels)
          
  def __getitem__(self,idx):  

    if isinstance(self.ds[idx][0], PIL.Image.Image): #depends on the transform used
      image = torchvision.tv_tensors.Image( to_tensor(self.ds[idx][0]) )
    else:
      image = torchvision.tv_tensors.Image(self.ds[idx][0])

    targetdic = { 'image_id': idx, 
    'iscrowd': torch.tensor([False]).repeat(self.bboxcoords[idx].shape[0]), 
    'boxes': copy.deepcopy(self.bboxcoords[idx]) ,
    'labels': self.clslabels[idx] , 
    'area': self.origareas[idx] 
    }

    if self.transforms is not None:
        image, targetdic = self.transforms(image, targetdic)

    return (image,targetdic)

def run():

  debugmaxnum = -1

  #https://pytorch.org/vision/stable/models.html  
  modelweights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
  model =   torchvision.models.detection.retinanet_resnet50_fpn_v2(weights = modelweights)

  #modelweights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
  #model =   torchvision.models.detection.ssd300_vgg16(weights = modelweights)
  
  #transforms = modelweights.transforms()
  vocdspath = '/home/binder/entwurf9/2024teaching/intro_computervision/somedatasets2/voc2007/voctrainval_06-nov-2007/'
  ds_val_pre  = torchvision.datasets.VOCDetection(root= vocdspath, year= '2007', image_set = 'val', transforms = None) #'train'  
  #print(ds_val_pre[0][1]['annotation'].keys())
  #print(ds_val_pre[0][1]['annotation']['size']) 
  #exit()
  
  transforms = None
  ds_val = torchvisionvoc_coco_bridge(ds_val_pre, transforms)
  print('done creating bridge class')
 
  ds_val_loader=  torch.utils.data.DataLoader(ds_val, batch_size= 1, shuffle=False )

  cocoAnnotation = get_coco_api_from_dataset(ds_val_loader.dataset)
  coco_evaluator2 = CocoEvaluator( cocoAnnotation , iou_types = ['bbox'])

  print('done creating cocoAnnotation and coco_evaluator2')
   
  model.eval()
  #model.to('cuda:0')

  n_threads = torch.get_num_threads()

  with torch.no_grad():
  
    for i, (img,target) in enumerate(ds_val_loader): 
   
      if (debugmaxnum > 0) and (i >= debugmaxnum): 
        break
        
        
      print('at', i)   
      print('img.shape', img.shape)
      print("target['boxes'].shape)", target['boxes'].shape)
      print("target['labels'].shape", target['labels'].shape)
                                       
      #img = img.to('cuda:0') 
      predictions = model(img)
      predicted = predictions[0] # unbatching

      print('num pred boxes', predicted['boxes'].shape[0])
      predsth = False
      if predicted['boxes'].shape[0]>0:  
        
        #TODO
        remappedlabels = ds_val.remaptensor_coco2voclabels_cpudevice(predicted['labels'].to('cpu'))
        #TODO        
        predsdic = { 
        "labels": remappedlabels, 
        'boxes': predicted['boxes'].to('cpu'),
        'scores': predicted['scores'].to('cpu'),        
         }
         
        res = { target["image_id"].item(): predsdic}
        coco_evaluator2.update(res)
        predsth = True
      
      # for testing of coco api with ground truth labels
      if 1==0:

        predsdic = { 
        'labels': target['labels'].squeeze(0),
        'boxes': target['boxes'].squeeze(0),
         'scores': torch.ones_like(target['labels'].squeeze(0)),
         }
        res = { target["image_id"].item(): predsdic}
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


    
