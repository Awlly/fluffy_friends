import torch
from PIL import Image
from .loader import transform, description, class_names, model



def classify_breed (image):

    image = Image.open(image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    _, predicted = torch.max(outputs, 1)
    predicted_class_name = class_names[predicted.item()]
    descr = str(description.loc[description['breed'] == predicted_class_name, 'care'].values)
    breed = (f"Я думаю, что это...\n{predicted_class_name}!\nВот краткое описание и советы по уходу для этой породы:\n{descr[2:-2]}")

    return breed

