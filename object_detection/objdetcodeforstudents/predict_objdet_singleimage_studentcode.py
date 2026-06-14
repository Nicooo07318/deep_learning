import PIL.Image
from torchvision.utils import draw_bounding_boxes

from torchvision.transforms.functional import to_pil_image
import torchvision.models.detection
import torch
import os

from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights


def run():

  

  filename = ['cows2-4301076_960_720', 'cows-5080091_960_720', 'morecows_4291970_960_720']
  

  #https://pytorch.org/vision/stable/models.html  
  modelweights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 # model weights

  model = retinanet_resnet50_fpn_v2(modelweights)  # your model, should use modelweights
  
  transforms = modelweights.transforms() # get transforms from modelweights
  for image_name in filename:
      
    # prepare batch made from one image
    img = PIL.Image.open("./testimages1/" + image_name + ".jpg").convert('RGB')  
    batch = [transforms(img)]

    # if you forget that, you will see a nice error :) 
    model.eval()

    with torch.no_grad():

      # call your model on the batch
      predictions = model(batch)
      
      predicted = predictions[0] # batch has size 1, see above, so we grab that one element here

      confidence_threshhold = 0.85

      mask = predicted["scores"] >= confidence_threshhold
      predicted["boxes"] = predicted["boxes"] [mask]
      predicted["scores"] = predicted["scores"] [mask]
      predicted["labels"] = predicted["labels"] [mask]
      
      #get names of labels which are predicted 
      labels_in_image = [modelweights.meta["categories"][i] for i in predicted["labels"]]

    # output what is predicted to understand the format of torchvision model outputs
    # it is a list of dictionaries. list has length equal to batchsize
    print(type(predictions))
    print(type(predicted))
    # each dictionary has keys for the outputs which one would expect from an object detection neural net
    for k in predicted.keys():
      print('dictionary key name {}'.format(k))  

    print(predicted)    
    print('unique predicted labels:')
    print( set(labels_in_image))

    # draw the boundingbox
    box = draw_bounding_boxes(batch[0], boxes=predicted["boxes"],
                              labels=labels_in_image,
                              colors="red",
                              width=4 ) #, font_size=30, font =None)
    #print convert to a PIL.Image                                                      
    im = to_pil_image(box.detach())
    #display the PIL.Image
    im.show()

    # -- Export predictions for evaluation --
    out_path = "./pred_testimages1/" + image_name + ".txt"
    all_preds = zip(predicted["boxes"].tolist(), predicted["scores"].tolist(), labels_in_image)
    print(all_preds)

    with open(out_path, "w") as f:
      for box, score, label_name in all_preds:
        print(box, score, label_name)
        f.write(f"{label_name} {score} {box[0]:.0f} {box[1]:.0f} {box[2]:.0f} {box[3]:.0f}\n")



  
  # with open(out_path, "w") as f:
  #   for box, score, label in all_preds:
  #     x1, y1, x2, y2 = box
  #     # <class_name> <confidence> <left> <top> <right> <bottom>
  #     f.write(label, score, x1, y1, x2, y2)
  #   print("Saved predictions to ", out_path)




if __name__=='__main__':
  run()  


    
