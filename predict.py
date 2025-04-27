import torch
from PIL import Image
import torchvision.transforms as transforms
from model import MyCustomModel
from config import resize_x, resize_y, save_path  # Use values from config

def cryptic_inf_f(image_path, model_weights_path=save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = MyCustomModel().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Transformation to match the input size for the model
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),  # Use resize values from config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])  # Adjust normalization
    ])
    
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict with the trained model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)  # Get the class with the highest probability
    
    return predicted.item()  # Return the predicted class as an integer

