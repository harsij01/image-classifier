import torch
from torchvision import models, transforms
from PIL import Image

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

categories = weights.meta["categories"]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('image.jpg').convert("RGB")

processed_img = preprocess(img)

batch = torch.unsqueeze(processed_img, 0)

with torch.no_grad():
    predictions = model(batch)

probabilities = torch.nn.functional.softmax(predictions, dim=1)[0] * 100

top_score, top_index = torch.max(probabilities, dim=0)

label = categories[top_index.item()]
confidence = top_score.item()
print(f"Prediction: {label} ({confidence:.2f}% confident)")

top5_score, top5_index = torch.topk(probabilities, 5)

for i in range(5):
    idx = top5_index[i].item()
    score = top5_score[i].item()
    print(f"{i + 1}. {categories[idx]}: {score:.2f}%")