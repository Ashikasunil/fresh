import torch
import torchvision.transforms as transforms
from model_arch import QRC_UNet

def load_model(model_path):
    model = QRC_UNet()
    model.load_state_dict(torch.load(best_mobilevit_qrc_unet.pth, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze(0).squeeze(0).numpy()
