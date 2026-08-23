from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title = "Text Summarizer App", description = "Text Summarization using T5", version = "1.0")

model = T5ForConditionalGeneration.from_pretrained("./save_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./save_summary_model")

if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print("device: ", device)
model.to(device)

templates = Jinja2Templates(directory = ".")

class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):

  text = re.sub(r"\r\n"," ",text)
  text = re.sub(r"\s+"," ",text)
  text = re.sub(r"<.*?>", " ", text)
  text = text.strip().lower()

  return text



def summarize_dialogue(dialogue: str) -> str:
  # clean data
  dialogue = clean_data(dialogue)

  # tokenize | inputs = dialogue tokens

  inputs = tokenizer(
      dialogue,
      padding = "max_length",
      max_length = 512,
      truncation = True,
      return_tensors = "pt"
  ).to(device)

  # generate dialogue summary: => token ids

  model.to(device)
  targets = model.generate(
  input_ids = inputs["input_ids"],
  attention_mask = inputs["attention_mask"],
  max_length = 150,
  num_beams = 4,             # => transformer will generate 4 different summaries and will give us the best among these
  early_stopping = True     # => as we get 4 summaries it will stop
  )


  # token ids convert to text => decoding

  summary = tokenizer.decode(targets[0], skip_special_tokens = True)    # => skip means skipping EOS,SEP,etc | we do not store them

  return summary

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"request": request}
)


