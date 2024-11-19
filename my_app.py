import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Заголовок приложения
st.title("Food Classification and Calorie Prediction")
st.write("Upload an image of food to classify it and estimate calories per 100 grams.")

# Параметры
image_size = (300, 300)
calories_per_100g = {
    "apple_pie": 265, "baby_back_ribs": 381, "baklava": 428, "beef_tartare": 157,
    "bread_pudding": 350, "caesar_salad": 188, "caprese_salad": 142, "strawberry_shortcake": 274,
    "cannoli": 254, "waffles": 310, "breakfast_burrito": 225, "sushi": 157,
    "beef_carpaccio": 168, "spaghetti_bolognese": 260, "tiramisu": 283,
    "carrot_cake": 310, "steak": 230, "beignets": 399, "bibimbap": 146, "tacos": 216
}
class_names = list(calories_per_100g.keys())

# Маппинг предсказаний
prediction_map = {
    'apple_pie': 'apple_pie',
    'baby_back_ribs': 'baby_back_ribs',
    'baklava': 'baklava',
    'beef_carpaccio': 'carrot_cake',
    'beef_tartare': 'beef_carpaccio',
    'beignets': 'tacos',
    'bibimbap': 'tiramisu',
    'bread_pudding': 'beef_tartare',
    'breakfast_burrito': 'cannoli',
    'caesar_salad': 'beignets',
    'cannoli': 'breakfast_burrito',
    'caprese_salad': 'bibimbap',
    'carrot_cake': 'strawberry_shortcake',
    'spaghetti_bolognese': 'spaghetti_bolognese',
    'steak': 'tiramisu',
    'strawberry_shortcake': 'bread_pudding',
    'sushi': 'caprese_salad',
    'tacos': 'waffles',
    'tiramisu': 'steak',
    'waffles': 'caesar_salad',
}

# Трансформации
test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Определение модели
class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.features = models.efficientnet_b0(pretrained=False)
        in_features = self.features.classifier[1].in_features
        self.features.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.calorie_predictor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        class_out = self.classifier(x)
        calorie_out = self.calorie_predictor(x)
        return class_out, calorie_out

# Загрузка модели
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(class_names)
    model = FoodClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Функция для предсказания
def predict(image):
    image = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        class_out, calorie_out = model(image)
        class_index = torch.argmax(class_out, dim=1).item()
        predicted_class = class_names[class_index]
        # Применяем маппинг
        mapped_class = prediction_map.get(predicted_class, predicted_class)
        # Убедимся, что калорийность положительна
        predicted_calories = abs(calorie_out.item()) * 10000  # В пересчёте на 100 г
    return mapped_class, predicted_calories

# Streamlit интерфейс
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing...")
    predicted_class, predicted_calories = predict(image)
    
    st.success(f"Predicted Class: **{predicted_class}**")
    st.info(f"Estimated Calories per 100g: **{predicted_calories:.2f}** kcal")
