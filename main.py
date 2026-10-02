import torch
from torchvision import models, transforms
import gradio as gr
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

def predict(img):
    if img is None:
        return {}

    processed_img = preprocess(img)
    batch = torch.unsqueeze(processed_img, 0)

    with torch.no_grad():
        predictions = model(batch)

    probabilities = torch.nn.functional.softmax(predictions, dim=1)[0] * 100

    top5_score, top5_index = torch.topk(probabilities, 5)
    results = {categories[top5_index[i].item()]: float(top5_score[i].item()) for i in range(5)}

    return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="ResNet-50 Image Classifier"
)

if __name__ == "__main__":
    demo.launch()