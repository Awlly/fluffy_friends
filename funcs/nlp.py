from sentence_transformers import SentenceTransformer, util
from .pet_type_detection import detect_pet_type
from .loader import cat_breeds_df, dog_breeds_df



from sentence_transformers import SentenceTransformer, util

def detect_breed(query):
    pet_type = detect_pet_type(query)
    if pet_type == 'cat':
        similarities = get_similarity_scores(query, cat_breeds_df)
        text = 'Наиболее подходящие под ваше описание породы кошек: \n\n'

        return str_top_matches(cat_breeds_df, similarities, text)
    elif pet_type == 'dog':
        similarities = get_similarity_scores(query, dog_breeds_df)
        text = 'Наиболее подходящие под ваше описание породы собак: \n\n'

        return str_top_matches(dog_breeds_df, similarities, text)

def get_similarity_scores(query, df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    descriptions = df['Full Description'].tolist()
    description_embeddings = model.encode(descriptions)
    similarities = util.pytorch_cos_sim(query_embedding, description_embeddings)

    return similarities

def str_top_matches(df, similarities, text):
    top_matches = similarities.argsort(descending=True)[0][:3]
    top_matches_list = top_matches.tolist()
    
    for idx in top_matches_list:
        breed_name = str(df.iloc[idx]['Официальное название'])
        breed_description = str(df.iloc[idx]['Описание'])
        text += f"{breed_name}\n\n{breed_description}\n\n"

    return text[:-1]