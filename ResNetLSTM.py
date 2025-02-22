import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2, face_recognition

#Model with feature visualization
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))
    
class ImagePredictor:
    def __init__(self, model, im_size=112, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.model = model
        self.im_size = im_size
        self.mean = mean
        self.std = std
        self.sm = nn.Softmax()
        self.inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std), std=np.divide([1,1,1],std))

    def im_convert(self, tensor):
        """ Convert tensor to image """
        image = tensor.to("cpu").clone().detach()
        image = image.squeeze()
        image = self.inv_normalize(image)
        image = image.numpy()
        image = image.transpose(1,2,0)
        image = image.clip(0, 1)
        cv2.imwrite('Output/original.png', image*255)
        return image

    def predict(self, img):
        """ Perform prediction on input image """
        fmap, logits = self.model(img.to('cuda'))
        weight_softmax = self.model.linear1.weight.detach().cpu().numpy()
        logits = self.sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        print('Confidence of prediction:', confidence)
        idx = np.argmax(logits.detach().cpu().numpy())
        bz, nc, h, w = fmap.shape
        out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx, :].T)
        predict = out.reshape(h, w)
        predict = predict - np.min(predict)
        predict_img = predict / np.max(predict)
        predict_img = np.uint8(255 * predict_img)
        out = cv2.resize(predict_img, (self.im_size, self.im_size))
        heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
        img = self.im_convert(img[:, -1, :, :, :])
        result = heatmap * 0.5 + img * 0.8 * 255
        cv2.imwrite('Output/predict.png', result)
        return [int(prediction.item()), confidence]

class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []      
        for i,frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top,right,bottom,left = faces[0]
                frame = frame[top:bottom,left:right,:]
            except:
                pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

"""
im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transforms = transforms.Compose([ transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

path_to_videos= []

video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
model = Model(2).cuda()
path_to_model = "C:\Final_Year_Project\Facial_Authentication\Models\Model_90_20_FF.pt"
model.load_state_dict(torch.load(path_to_model))
model.eval()
img_pred = ImagePredictor(model)
for i in range(0,len(path_to_videos)):
    print(path_to_videos[i])
    prediction = img_pred.predict(video_dataset[i])
    print(prediction)
    if prediction[0] == 1:
        print("REAL")
    else:
        print("FAKE")
"""
