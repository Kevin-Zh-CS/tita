import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import gc
from torch.cuda.amp import autocast



BIG_MODEL_NAME = "PATH_TO_PRETRAINED_MODEL"    
SMALL_MODEL_NAME = "PATH_TO_REWARD_MODEL"  
DATASET_NAME = "MM-Vet/mm-vet"
WEIGHT_BIG = 0.7
WEIGHT_SMALL = 0.3
DEVICE = "cuda"


big_model = AutoModelForVision2Seq.from_pretrained(BIG_MODEL_NAME, torch_dtype=torch.float16)
big_processor = AutoProcessor.from_pretrained(BIG_MODEL_NAME)

small_model = AutoModelForVision2Seq.from_pretrained(SMALL_MODEL_NAME, torch_dtype=torch.float16)
small_processor = AutoProcessor.from_pretrained(SMALL_MODEL_NAME)


dataset = load_dataset("MM-Vet/mm-vet", split="validation")


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image_from_url_or_path(path_or_url):
    try:
        return Image.open(path_or_url).convert("RGB")
    except Exception:
        return None


def fused_inference(image, question):
    try:
        big_model.to(DEVICE)
        inputs_big = big_processor(images=image, text=question, return_tensors="pt").to(DEVICE, torch.float16)
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            out_big = big_model(**inputs_big, return_dict=True)
        logits_big = out_big.logits[:, -1, :].cpu()
        big_model.to("cpu")
        del inputs_big, out_big
        torch.cuda.empty_cache()

        small_model.to(DEVICE)
        inputs_small = small_processor(images=image, text=question, return_tensors="pt").to(DEVICE, torch.float16)
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            out_small = small_model(**inputs_small, return_dict=True)
        logits_small = out_small.logits[:, -1, :].cpu()
        small_model.to("cpu")
        del inputs_small, out_small
        torch.cuda.empty_cache()

        fused_logits = WEIGHT_BIG * logits_big + WEIGHT_SMALL * logits_small
        top_token_id = torch.argmax(fused_logits, dim=-1)
        answer = big_processor.tokenizer.decode(top_token_id[0], skip_special_tokens=True)
        return answer.strip()

    except Exception as e:
        return f"[ERROR] {str(e)}"
    finally:
        gc.collect()
        torch.cuda.empty_cache()


correct = 0
total = 0

for item in dataset:
    image = load_image_from_url_or_path(item["image_path"])
    if image is None:
        continue

    question = item["question"]
    gold_answer = item["answer"]

    pred = fused_inference(image, question)
    print(f"Q: {question}")
    print(f"Predicted: {pred}")
    print(f"Gold: {gold_answer}")
    print("-" * 50)

    if pred.lower() in [a.lower() for a in gold_answer]:
        correct += 1
    total += 1

acc = correct / total if total else 0
print(f"\n✅ Fused Accuracy: {acc * 100:.2f}% on {total} samples")