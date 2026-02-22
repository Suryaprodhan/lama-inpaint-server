import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import io
import uvicorn

app = FastAPI()

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    # ইমেজ এবং মাস্ক রিড করা
    contents = await image.read()
    mask_contents = await mask.read()

    # OpenCV ফরম্যাটে কনভার্ট করা
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    mask_nparr = np.frombuffer(mask_contents, np.uint8)
    mask_img = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)

    # মাস্ক যদি ইমেজের সাইজের না হয়, তবে রিসাইজ করা
    mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]))

    # অবজেক্ট রিমুভালের আসল লজিক (OpenCV Inpainting)
    # এখানে Telea বা Navier-Stokes অ্যালগরিদম ব্যবহার করা হয়
    result = cv2.inpaint(img, mask_img, 3, cv2.INPAINT_TELEA)

    # রেজাল্ট সেভ করা
    save_path = "/tmp/result.png"
    cv2.imwrite(save_path, result)
    
    return FileResponse(save_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
