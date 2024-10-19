import torch
from torch import nn
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class Conv_block(nn.Module):
    def __init__(self, in_channels,out_channels, use_1x1conv=False,stride=1,padding = 1):
        super(Conv_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.ibn1 = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.ibn1(x)
        x = F.relu(x)
        return x
class ca2_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(ca2_Residual, self).__init__()

        self.Conv_block1 = Conv_block(in_channels=in_channels, out_channels=out_channels, stride=1,padding=1)
        self.Conv_block2 = Conv_block(in_channels=out_channels, out_channels=out_channels, stride=1,padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1)
        self.ibn_3 = nn.BatchNorm1d(out_channels)

        self.conv1_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.cbam = CBAM(in_channels=out_channels)
        
    def forward(self, x):
        y = self.Conv_block1(x)
        y = self.Conv_block2(y)
        y = self.ibn_3(self.conv1d_3(y))
        y = self.cbam(y)
        x = self.conv1_1(x)
        y =y+x
        end = F.relu(y)
        return end 
class ca2_resnet(nn.Module):
    def __init__(self):
        super(ca2_resnet, self).__init__()
        self.conv1 = Conv_block(in_channels=3, out_channels=8, stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3,stride=1)

        self.caResnet1 = ca2_Residual(in_channels=8, out_channels=16, stride=1)
        self.caResnet2 = ca2_Residual(in_channels=16, out_channels=32, stride=1)
        self.caResnet3 = ca2_Residual(in_channels=32, out_channels=32, stride=1)
        self.caResnet4 = ca2_Residual(in_channels=32, out_channels=32, stride=1)

        self.Fl = nn.Flatten()
        self.mlp1 = nn.Sequential(
            nn.Linear(32*98, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            #nn.Softmax(dim=1)
            nn.Sigmoid()


        )
        self.mlp2 = nn.Sequential(
            nn.Linear(32*98, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256,2),

        )
        self.mlp3 = nn.Sequential(
            nn.Linear(32*98, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256,2),

        )
        
    def forward(self, x):
  
        x = self.conv1(x)#
        x = self.maxpool1(x)
        x = self.caResnet1(x)
        x = self.caResnet2(x)
        x = self.caResnet3(x)
        x = self.caResnet4(x)
        x = self.Fl(x)
        x1 = self.mlp1(x)
        x2 = self.mlp2(x)
        x3 = self.mlp3(x)


        return x1,x2,x3
        
def test(model, device, test_loader, threshold):
    results = {'targets': [], 'preds': [], 'outputs': [], 'pre_ew1': [], 'pre_ew2': [], 'pre_fwhm1': [], 'pre_fwhm2': []}
    
    model.eval()
    with torch.no_grad():
        for feature, label in test_loader:
            feature, label_class = feature.to(device), label[:, 0:1].to(device)
            outputs1, outputs2, outputs3 = model(feature)
            
            results['preds'].extend((outputs1 >= threshold).cpu().numpy())
            results['targets'].extend(label_class.cpu().numpy())
            results['outputs'].extend(outputs1.cpu().numpy())
            results['pre_ew1'].extend(outputs2[:, 0:1].cpu().numpy())
            results['pre_ew2'].extend(outputs2[:, 1:2].cpu().numpy())
            results['pre_fwhm1'].extend(outputs3[:, 0:1].cpu().numpy())
            results['pre_fwhm2'].extend(outputs3[:, 1:2].cpu().numpy())
    
    return pd.DataFrame({key: np.ravel(values) for key, values in results.items()})

batch_size = 1

pd_test = pd.read_csv(r"TestData_input.csv").to_numpy()
df_test_x = pd_test[:, :300].reshape(-1, 3, 100)
df_test_y = np.ones((len(df_test_x), 1))

x_test = torch.tensor(df_test_x, dtype=torch.float32)
y_test = torch.tensor(df_test_y, dtype=torch.float32)

test_loader = data.DataLoader(data.TensorDataset(x_test, y_test), batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = ca2_resnet().to(device)
net.load_state_dict(torch.load(r"ResNet_CBAM.pth"), strict=False)

print("\nResult:")
test_results = test(net, device, test_loader, threshold=0.50)
correct_count = len(test_results[(test_results['targets'] == 1) & (test_results['outputs'] > 0.5)])

print("Number of positive samples:", len(pd_test))
print("Number of positive samples correctly predicted:", correct_count)
