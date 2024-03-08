import torch
from torchvision import  models
import pandas as pd
import torchvision.models as models
from torchvision import transforms



def load_data():

    #NLP data
    cat_breeds_df = pd.read_csv('data/cats.csv')
    dog_breeds_df = pd.read_csv('data/dogs.csv')
    cat_breeds_df['Full Description'] = cat_breeds_df[cat_breeds_df.columns.difference(['Официальное название'])].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1
    )
    dog_breeds_df['Full Description'] = dog_breeds_df[dog_breeds_df.columns.difference(['Официальное название'])].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1
    )

    #Transforms
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #CV data
    description = pd.read_csv ("data/breed_descriptions.csv")
    description.drop('Id', axis=1, inplace=True)
    class_names = description['breed'].unique().tolist()

    #CV model
    device = torch.device("cpu")
    num_classes = 1000
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('data/breeds_classification_model.pth'))
    model.to(device)
    model.eval()

    return model, description, class_names, device, cat_breeds_df, dog_breeds_df, transform

model, description, class_names, device, cat_breeds_df, dog_breeds_df, transform = load_data()