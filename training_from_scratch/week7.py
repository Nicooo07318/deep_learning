import torch 
import torchvision as tv
import PIL.Image 
import os
import time
import io

from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

class FlowersDataset(Dataset):
    def __init__(self, split_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = [] # tuple: (imagename, label)
        
        with open(split_file, "r") as f:
            lines = f.readlines()

            for l in lines:
                parts = l.strip().split()
                img_name = parts[0]
                label = int(parts[1]) # -1 wenn labels mit 1 anfangen
                self.samples.append((img_name, label))

        # pre-loading in RAM
        print("Caching images...")
        self.images = []
        for img_name, _ in self.samples:
            img_path = os.path.join(self.image_dir, img_name)
            with open(img_path, 'rb') as f:
                self.images.append(f.read())
        print("Done Caching")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # img_name, label = self.samples[index]
        # img_path = os.path.join(data, img_name)

        # image_obj = PIL.Image.open(img_path).convert('RGB')

        img_bytes = self.images[index]
        image_obj = PIL.Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.samples[index][1]

        if self.transform:
            image_obj = self.transform(image_obj)
        
        return image_obj, label


train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def train_epoch(model,  trainloader,  criterion, device, optimizer ):

    
    t0 = time.time()

    model.train() # IMPORTANT!!!
 
    losses = []
    for batch_idx, data in enumerate(trainloader):

        t_data = time.time()

        inputs=data[0].to(device)
        labels=data[1].to(device) 

        # print("inputs on device: ", inputs.device)
        # print("labels on device: ", labels.device)
        
        t_data_done = time.time()
       
        optimizer.zero_grad() #reset accumulated gradients 

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

         
        loss.backward() #compute new gradients
        optimizer.step() # apply new gradients to change model parameters

        t_gpu_done = time.time()
       
        # if batch_idx % 10 == 0:
        #     print(f"Batch {batch_idx} | data -> Gpu: {t_data_done-t_data:.3f}s | forward+backward: {t_gpu_done-t_data_done:.3f}s")


    #print(losses)

    return losses


def evaluate(model, dataloader, criterion, device):

    model.eval() # IMPORTANT!!!


    with torch.no_grad(): # do not record computations for computing the gradient
    
      datasize = 0
      accuracy = 0
      avgloss = 0
      total_loss = 0.0
      total_samples = 0
      total_correct = 0
      for ctr, data in enumerate(dataloader):

          #print ('epoch at',len(dataloader.dataset), ctr)
          
          inputs = data[0].to(device)
          labels = data[1].to(device)        
          outputs = model(inputs)

          #accuracy
          _, preds = torch.max(outputs, 1)
          total_correct += torch.sum(preds == labels).item()
          total_samples += labels.shape[0]

          # computing some loss
          if criterion is not None:

            #loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.shape[0]

            

    
    avgloss = (total_loss / total_samples) if criterion is not None else None
    accuracy = total_correct / total_samples
    
    
          
    return accuracy, avgloss

def train_modelcv(dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device):

  best_acc = 0
  best_epoch =-1

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    training_losses=train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )
    #scheduler.step()
    measured_acc, measured_loss = evaluate(model, dataloader_cvtest, criterion = None, device = device)
    
    print('perfmeasure acc: ', measured_acc)

    if measured_acc > best_acc:
        bestweights = model.state_dict()
        best_acc = measured_acc
        best_epoch = epoch
        print('current best', measured_acc, ' at epoch ', best_epoch)
        loss_at_best_acc = measured_loss
    

  return best_epoch, best_acc, bestweights, loss_at_best_acc

def mode_a(model):
    model = tv.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.01, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    a_best_epoch, a_best_acc, a_bestweights, a_loss = train_modelcv(
        dataloader_cvtrain = train_dataloader,
        dataloader_cvtest = val_dataloader,  
        model = model,  
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = None, 
        num_epochs = maxnumepochs, 
        device = device)

    return a_best_epoch, a_best_acc, a_bestweights, a_loss

def mode_b(model, weights):
    weights = ResNet50_Weights.DEFAULT
    model = tv.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    b_best_epoch, b_best_acc, b_bestweights, b_loss = train_modelcv(
        dataloader_cvtrain = train_dataloader,
        dataloader_cvtest = val_dataloader,  
        model = model,  
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = None, 
        num_epochs = maxnumepochs, 
        device = device)

    return b_best_epoch, b_best_acc, b_bestweights, b_loss

def mode_c(model, weights):
    weights = ResNet50_Weights.DEFAULT
    model = tv.models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)

    for param in model.parameters():
        param.requires_grad = False # freeze everything

    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr= 0.001, 
        momentum=0.9)
        
    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    c_best_epoch, c_best_acc, c_bestweights, c_loss = train_modelcv(
        dataloader_cvtrain = train_dataloader,
        dataloader_cvtest = val_dataloader,  
        model = model,  
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = None, 
        num_epochs = maxnumepochs, 
        device = device)

    return c_best_epoch, c_best_acc, c_bestweights, c_loss

def run():


    

    weights = ResNet50_Weights.DEFAULT
    model_scratch = tv.models.resnet50(weights=None)
    model = tv.models.resnet50(weights=weights)


    a_best_epoch, a_best_acc, a_bestweights, a_loss = mode_a(model=model_scratch)
    b_best_epoch, b_best_acc, b_bestweights, b_loss = mode_b(model, weights)
    c_best_epoch, c_best_acc, c_bestweights, c_loss = mode_c(model, weights)


    results = {
        'a': (a_best_acc, a_bestweights),
        'b': (b_best_acc, b_bestweights),
        'c': (c_best_acc, c_bestweights),
    }

    best_mode =max(results, key=lambda k: results[k][0])
    best_acc, best_weights = results[best_mode]

    print("best model: ", best_mode.upper(), 'mit acc: ', best_acc)


    if best_mode == 'a':
        best_model = tv.models.resnet50(weights=None)
    elif best_mode in ('b', 'c'):
        best_model = tv.models.resnet50(weights=ResNet50_Weights.DEFAULT)


    best_model.load_state_dict(best_weights)
    best_model.eval()

    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        test_loss, test_acc = evaluate(best_model, test_dataloader, criterion, device)
        print('Test Accuracy: ', test_acc)
        print('Test Loss´: ', test_loss)
    



    # plot data:



if __name__ == '__main__':

    # global variables

    work_dir = os.getcwd()
    data = os.path.join(work_dir, 'flowers_data/jpg')
    train_file = os.path.join(work_dir, 'flowers_data/trainfile.txt')
    test_file = os.path.join(work_dir, 'flowers_data/testfile.txt')
    val_file = os.path.join(work_dir, 'flowers_data/valfile.txt')

    maxnumepochs = 5


    




    train_dataset = FlowersDataset(train_file, data, transform=train_transform)
    test_dataset = FlowersDataset(test_file, data, val_transform)
    val_dataset = FlowersDataset(val_file, data, val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32 , shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, 32, False, num_workers=2) 
    val_dataloader = DataLoader(val_dataset, 32, num_workers=2)

    # run programm
    run()