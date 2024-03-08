from sentence_transformers import SentenceTransformer, util



def detect_pet_type(query):
    cat_statements = [
        "Коты любят мурлыкать и играть одни.",
        "Хочу котёнка, который будет лежать на моих коленях.",
        "Кошки независимы и любят чистоту.",
        "Ищу кота с мягкой шерстью и острыми когтями.",
        "Кошка, которая любит смотреть в окно."
    ]
    dog_statements = [
        "Собаки любят играть на улице и нуждаются в прогулках.",
        "Мечтаю о верном друге, который будет защищать меня.",
        "Щенок, который будет радоваться и прыгать, увидев меня.",
        "Хочу собаку, которая любит воду и приключения.",
        "Пёс, который будет следовать за мной повсюду."
    ]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode(query)
    cat_embeddings = model.encode(cat_statements)
    dog_embeddings = model.encode(dog_statements)

    cat_similarity = util.pytorch_cos_sim(query_embedding, cat_embeddings).mean()
    dog_similarity = util.pytorch_cos_sim(query_embedding, dog_embeddings).mean()

    similarity_threshold = 0.6
    max_similarity = max(cat_similarity.item(), dog_similarity.item())

    if max_similarity < similarity_threshold:
        return 'unknown'
    elif cat_similarity > dog_similarity:
        return 'cat'
    elif dog_similarity > cat_similarity:
        return 'dog'