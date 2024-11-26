import transformers
from  llama_recipes.inference.model_utils import load_model
from transformers import AutoTokenizer
import time 
from accelerate.utils import is_xpu_available
import torch

instr = "In the following paragraph of text, some of the words have been replaced by the word XXXX. Please reply with the exact same paragraph word for word, but replacing the word XXXX with the best possible word to ensure that the paragraph makes sense. Change nothing else, except for the word XXXX. "
sampl = "Wells Cathedral is an Anglican place of XXXX in Wells, Somerset, dedicated to XXXX the Apostle. It is the seat of the Bishop of Bath and Wells, whose cathedra it holds as mother XXXX of the Diocese of Bath and Wells. Built as a XXXX Catholic cathedral from around 1175 to replace an earlier XXXX on the site since 705, it became an Anglican XXXX when Henry VIII split from Rome. Its Gothic architecture is mostly inspired from Early English style of the late 12th to early 13th centuries. The stonework of its pointed arcades and fluted piers bears pronounced mouldings and carved capitals in a foliate, 'stiff-leaf' style. The east end retains much ancient stained glass. Unlike many cathedrals of monastic foundation, Wells has many surviving secular buildings linked to its chapter of secular canons, including the Bishop's Palace and the 15th-century residential Vicars' Close."

model_name = "Llama3.1-70B-Instruct"
model_path = "/work/pcsl/model_weights/llama_hf_weights/"
# quantization = "4bit"
# quantization = "8bit"
quantization = None 
model = load_model(model_path+model_name, quantization, True,model_type = "llama")
model.eval() #put the modeli neval mode.
tokenizer = AutoTokenizer.from_pretrained(model_path+model_name)
tokenizer.pad_token = tokenizer.eos_token

max_padding_length=None
do_sample = True
min_length=None 
use_cache=True 
repetition_penalty=1.0 
length_penalty = 1.0 

def inference(
        user_prompt,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        **kwargs,
    ):
    batch = tokenizer(
            user_prompt,
            truncation=True,
            max_length=max_padding_length,
            return_tensors="pt",
        )
    if is_xpu_available():
        batch = {k: v.to("xpu") for k, v in batch.items()}
    else:
        batch = {k: v.to("cuda") for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs,
        )
    e2e_inference_time = (time.perf_counter() - start) * 1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text 

prompt = instr + "\n\n" + sampl

unmasked_text = inference(prompt,\
          temperature=0.4,\
          top_p = 1.0,\
          top_k = 50,\
          max_new_tokens=400
          )
import pickle
fname = 'test-70b'+str(quantization)+'.txt'
with open(fname,'wb') as fh:
    pickle.dump(unmasked_text,fh)

print(unmasked_text)
