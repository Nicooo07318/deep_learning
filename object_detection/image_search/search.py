import os
import torch
import open_clip
from PIL import Image
import numpy as np
from pathlib import Path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k') 
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.to(device) # put model on GPU
model.eval() # turn on eval mode


# get all image paths inside a List
IMAGE_DIR = os.path.join("data", "val2017")

image_paths = [
    os.path.join(IMAGE_DIR, file)
    for file in os.listdir(IMAGE_DIR)
] 


# --- puts images through the NN --- 

@torch.no_grad() # no gradient tracking, bcs no training (backprop)
def embed_imgs(paths, batch_size=32):
    embeddings = []
    for i in range(0, len(paths), batch_size):# step size of 32
        batch_paths = paths[i : i+batch_size] # list contains 32 paths
        imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
        if i < 10: print("shape: ", imgs[0].shape) # for debugging / understanding
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch) # NN on the images
        feats = feats / feats.norm(dim=-1, keepdim=True) # normalize features
        if i < 10: print("vectors: ", feats) # for debugging / understanding
        embeddings.append(feats)
        print(f"Embedded {i + len(batch_paths)} / {len(paths)}")
    return torch.cat(embeddings, dim=0)





# --- puts the query through the NN ---
@torch.no_grad()
def embed_text(query):
    tokens = tokenizer([query]).to(device) # splits query into words (whitespaces)
    print("Tokens: ", tokens)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()

def search(query, top_result=5):
    data = torch.load("processed_imgs.pt")
    paths = data["paths"]
    embeddings = data["embeddings"]

    text_emb = embed_text(query)
    sims = (embeddings @ text_emb.T).squeeze(1)
    # print("shape: ", sims.shape)
    top_k = sims.topk(top_result).indices

    results = [(paths[i], sims[i].item()) for i in top_k]
    return results


if __name__ == '__main__':
    if not os.path.exists("processed_imgs.pt"):
        image_embeddings = embed_imgs(image_paths)
        torch.save({"paths": image_paths, "embeddings": image_embeddings}, "processed_imgs.pt") # save results from the nn with imagepath so search is faster
        

    results = search("a cat walking in the snow")
    for path, score in results:
        print(f"{score:.3f} {path}")
        img = Image.open(path)
        img.show()
        


