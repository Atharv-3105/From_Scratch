import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), #Shape [Batch, in_channels, img_height,img_width]----->[Batch, out_channels, img_height,img_width]
            nn.BatchNorm2d(out_channels), #It will normalize across the channels
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), #Shape [Batch, out_channels, img_height,img_width]----->[Batch, out_channels, img_height,img_width]
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    #out_channels = 1 because we are doing binary segmentation
    def __init__(self, in_channels=3, out_channels = 1, feature_dims = [64, 128, 256, 512]):
        super().__init__()
        self.up_sample = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        #Down Sample part
        for feature in feature_dims:
            self.down_sample.append(DoubleConv(in_channels, feature)) #It will map the in_channels to feature_dims
            in_channels = feature #in_channel dim will be updated to the feature_dimension {for eg:- in_channel = 64, feature=128 :- update in dim of in_channel = 128}
            
        #UP Sample Part
        for feature in reversed(feature_dims):
            #Shape [Batch, features*2, img_height, img_width]-------->[Batch, features, img_height*2, img_width*2]
            self.up_sample.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2)) 
            
            #Append DoubleConv block as while Up_Sampling we perform Expand Spatial Dimension-->2 Convolutions
            self.up_sample.append(DoubleConv(feature*2, feature))
            
        #Add the bottleneck layer
        self.bottleneck = DoubleConv(feature_dims[-1], feature_dims[-1]*2)  #Shape [Batch, 512, _reduced_img_height, reduced_img_width]-------->[Batch, 1024, reduced_img_height, reduced_img_width]
        
        #Final Convolution Layer to map the features to the num_classes
        self.final_output = nn.Conv2d(in_channels=feature_dims[0], out_channels=out_channels, kernel_size=1) #Shape [Batch, 64, original_img_height, original_img_width]-------->[Batch, 1, original_img_height, original_img_width] 
    
    
    def forward(self, x):
        skip_connections = []
        
        #----------------Broadcast the input img through the DownSampling phase----------------
        for down in self.down_sample:
            
            #We perform one operation of down_sampling phase
            x = down(x) 
            
            #We store the result of the operation performed in the skip_connection to use it while UpSampling to preserve the features which can get lost while performing operations
            skip_connections.append(x) 
            
            #Now perform MaxPooling 
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        #We reverse the skip_connections as we want the latest stored result 
        skip_connections = skip_connections[::-1]
        
        #----------------Broadcast the img through the UpSampling phase----------------
        
        #We are incrementing idx by 2 as we perform DoubleConvolutions
        for idx in range(0, len(self.up_sample), 2):
            x = self.up_sample[idx](x)
            
            #Concatenate the skip_connection result of the Down_sampling phase with the result of the up_sampling phase so that no feature is lost
            skip_connection = skip_connections[idx//2]
            
            #Fix for If the shapes mismatch of the input and the skip_connection at any step
            if x.shape != skip_connection.shape:
                #We resize the input to match the shape of the skip_connection {only the heigh , width}
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            skip_connection_result = torch.concat((skip_connection, x ), dim=1) #dim=1 because we are concatenating along the channels dimension which is at 1 
            x = self.up_sample[idx+1](skip_connection_result)
            
        return self.final_output(x)
    
#=======================function to check if the dimension of our model output is withheld throughout the operation or not=======================
def test():
    x = torch.rand((3,1,161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print("*"*15, f"The shape of model_output:-{preds.shape} == The shape of model_input:- {x.shape}", "*"*15)
    assert x.shape == preds.shape
    
    
if __name__ == "__main__":
    test()
        