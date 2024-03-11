from huggingface_hub import hf_hub_download
from gensim.models.fasttext import load_facebook_model

# download model from huggingface hub
model_path = hf_hub_download(repo_id="simonschoe/call2vec", filename="model.bin")

# load model via gensim
model = load_facebook_model(model_path)

# extract word embeddings
model.wv['transformation']
# get similar phrases
model.wv.most_similar(positive='transformation', topn=5)
# get dissimilar phrases
model.wv.most_similar(negative='transformation', topn=5, restrict_vocab=None)
# compute pairwise similarity scores (distance = 1 - similarity)
model.wv.similarity('transformation', 'continuity')
