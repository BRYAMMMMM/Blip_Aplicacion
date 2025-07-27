from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from io import BytesIO
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_1_PATH = BASE_DIR / "models" / "blip_model_1"
MODEL_2_PATH = BASE_DIR / "models" / "blip_model_2"

models = {
    "blip1": BlipForConditionalGeneration.from_pretrained(str(MODEL_1_PATH)),
    "blip2": BlipForConditionalGeneration.from_pretrained(str(MODEL_2_PATH))
}

processors = {
    "blip1": BlipProcessor.from_pretrained(str(MODEL_1_PATH)),
    "blip2": BlipProcessor.from_pretrained(str(MODEL_2_PATH))
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
for model in models.values():
    model.to(DEVICE)

@app.get("/models")
async def get_models():
    return {"models": list(models.keys())}

@app.post("/generate_caption/")
async def generate_caption(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in models:
        return {"error": "Modelo no encontrado"}

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    processor = processors[model_name]
    model = models[model_name]

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=300,
        num_beams=5,
        repetition_penalty=1.2
    )
    caption = processor.decode(out[0], skip_special_tokens=True)

    return {"caption": caption}
