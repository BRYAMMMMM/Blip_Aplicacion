from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
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
BLIP1_PATH = BASE_DIR.parent / "models" / "blip1" / "blip1-esp"
BLIP2_PATH = BASE_DIR.parent / "models" / "blip2" / "blip2-opt-2.7b"
LORA_PATH = BASE_DIR.parent / "models" / "blip2" / "blip2-lora-finetuned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

print("\nCargando BLIP1...")
blip1_model = BlipForConditionalGeneration.from_pretrained(
    str(BLIP1_PATH), local_files_only=True
).to(DEVICE)
blip1_processor = BlipProcessor.from_pretrained(str(BLIP1_PATH), local_files_only=True)

print("\nCargando BLIP2...")
blip2_base = Blip2ForConditionalGeneration.from_pretrained(
    str(BLIP2_PATH),
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
blip2_model = PeftModel.from_pretrained(blip2_base, str(LORA_PATH)).to(DEVICE)
blip2_processor = Blip2Processor.from_pretrained(
    str(BLIP2_PATH), local_files_only=True, trust_remote_code=True
)

models = {
    "blip1": {"model": blip1_model, "processor": blip1_processor},
    "blip2": {"model": blip2_model, "processor": blip2_processor},
}

print("\nModelos BLIP1 y BLIP2 cargados correctamente.\n")

@app.get("/models")
async def get_models():
    return {"models": list(models.keys())}

@app.post("/generate_caption/")
async def generate_caption(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in models:
        return {"error": f"Modelo '{model_name}' no disponible."}

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    processor = models[model_name]["processor"]
    model = models[model_name]["model"]

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3
        )
    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return {"caption": caption}

@app.get("/")
async def root():
    return {"message": "API funcionando con BLIP1 y BLIP2."}
