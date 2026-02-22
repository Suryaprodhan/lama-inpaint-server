# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import torch
import io
import os

# Assuming model weights are in 'checkpoints/big-lama.pt'
# For simplicity, we will simulate the inpainting logic
# (Because phone-based deploy CPU-only will be slow)

app = FastAPI()

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    mask_img = Image.open(mask.file).convert("L")

    # Convert to numpy
    img_np = np.array(img)
    mask_np = np.array(mask_img)

    # Fake inpainting: just fill mask with average color
    img_np[mask_np>128] = img_np[mask_np<=128].mean(axis=(0,1))

    out_img = Image.fromarray(img_np)
    out_img.save("result.png")
    return FileResponse("result.png")
