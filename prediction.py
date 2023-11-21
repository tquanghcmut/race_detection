import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# Define the transformations (make sure it's the same as used during training)
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2470, 0.2435, 0.2616])
])

# Define the classes
classes = ['Japanese', 'Other', 'Vietnamese']
class Vgg_m_face_bn_dag(nn.Module):

    def __init__(self):
        super(Vgg_m_face_bn_dag, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
        self.softmax8 = nn.Softmax(dim=1)
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24_preflatten = self.relu7(x23)
        x24 = x24_preflatten.view(x24_preflatten.size(0), -1)
        x25 = self.fc8(x24)
        x26 = self.softmax8(x25)
        return x26

def vgg_m_face_bn_dag(weights_path=None, **kwargs):
    model = Vgg_m_face_bn_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

VGG = vgg_m_face_bn_dag()
num_classes = 3
VGG.fc8 = nn.Linear(VGG.fc8.in_features, num_classes)
model_path = '/Users/twang/PycharmProjects/race_detection_official/model/VGG_augmentation_weight_class_norm.pt'
mps_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VGG.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
VGG.to(mps_device)
VGG.eval()

predictions_counter = {classname: 0 for classname in classes}
image_dir = '/Users/twang/PycharmProjects/race_detection_official/test/input'
save_dir = '/Users/twang/PycharmProjects/race_detection_official/test/output'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    if os.path.isfile(image_path):
        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        image_transformed = transformations(image).unsqueeze(0).to(mps_device)

        # Predict the class
        with torch.no_grad():
            outputs = VGG(image_transformed)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]
            # Increment the predictions counter
            predictions_counter[predicted_class] += 1

        # Draw the label on the image
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), predicted_class, (255, 0, 0), font=font)

        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Turn off the axis
        plt.show()

        # Save the newly annotated image
        save_path = os.path.join(save_dir, image_file)
        image.save(save_path)

# After all images are processed, print the predictions count
print("Prediction counts:")
for classname, count in predictions_counter.items():
    print(f"{classname}: {count}")

# Optional: Save the predictions count to a text file
with open(os.path.join(save_dir, 'predictions_count.txt'), 'w') as f:
    for classname, count in predictions_counter.items():
        f.write(f"{classname}: {count}\n")