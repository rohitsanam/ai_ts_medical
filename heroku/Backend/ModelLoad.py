import torch
import torchvision.models as models
import torch.nn as nn

# Define MedicalDenseNet (must match training architecture)
class MedicalDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(MedicalDenseNet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load Model Weights

MODEL_PATH = r"C:\Users\malia\OneDrive\Desktop\LungDisease\Models\densenet_final_model.pt"
num_classes = 5  # Match the number of classes in training
model = MedicalDenseNet(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()  # Set to inference mode

# Test Model Loading
print("Model loaded successfully!")