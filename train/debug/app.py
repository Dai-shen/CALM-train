from typing import Dict, List, Tuple, Union
from peft import PeftModel
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
from flask import Flask, render_template, request
import sys
sys.path.append("../src")


def load_model():
    model_name_or_path = '/nfs/a100-80G-17/jiyunjie/finetuned_ckpt/on_belle_tokenizer50k_openinstr_zh/zh_alpaca_gpt3.5_gpt4_sharegpt_epoch=2-step=20652'
    ckpt_path = '/nfs/a100-006/hanweiguang/saved_model/boxue_1_bs-8_lr-3e-4_wm-1e-2_epoch-10_lora'

    print('Loading model...')
    # Initialize the model and tokenizer
    load_type = torch.float16
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=load_type
    )
    model = PeftModel.from_pretrained(
        base_model, ckpt_path, torch_dtype=load_type
    )
    model.eval()
    model.to('cuda:0')
    print('Model loaded!')
    return model, tokenizer

# Initialize flask
app = Flask(__name__)

model, tokenizer = None, None


@app.before_first_request
def initialize_model():
    global model, tokenizer
    model, tokenizer = load_model()


def get_probability_per_token(input_ids: torch.Tensor) -> Tuple[List[float], float]:
    """
    input_ids: [1, sen_len]
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits
        probs = logits.softmax(dim=-1)
    # Select the probabilities of the generated tokens
    # [1, sen_len, 1]
    generated_token_probs = torch.gather(
        probs[:, :-1, :], 2, input_ids[:, 1:, None])  # skip the first token_id
    return generated_token_probs[0, ..., 0].tolist(), outputs.loss.item()


def decode_and_get_probability(input_text: str, generation_config: GenerationConfig):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
        getattr(model, 'module', model).device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids,
                                generation_config=generation_config)
    token_ids = output[0].tolist()
    tokens = [tokenizer.decode([token_id], skip_special_tokens=False)
              for token_id in token_ids]
    token_probs_list, loss = get_probability_per_token(output)

    # The probability of <s> is set to 0
    tokens_with_probs = [{'token': tokens[0], 'prob': 0.0}]
    for token, prob in zip(tokens[1:], token_probs_list):
        tokens_with_probs.append({
            'token': token,
            'prob': prob
        })

    return tokens_with_probs, loss


def get_probability_of_text(input_text: str) -> List[Dict[str, Union[float, int]]]:
    input_ids = tokenizer.encode(
        input_text, return_tensors='pt').to(
        getattr(model, 'module', model).device
    )
    token_ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([token_id], skip_special_tokens=False)
              for token_id in token_ids]
    token_probs_list, loss = get_probability_per_token(input_ids)

    # The probability of <s> is set to 0
    tokens_with_probs = [{'token': tokens[0], 'prob': 0.0}]
    for token, prob in zip(tokens[1:], token_probs_list):
        tokens_with_probs.append({
            'token': token,
            'prob': prob
        })

    return tokens_with_probs, loss


@app.route("/inference", methods=['POST'])
def inference():
    input_text = request.form.get('input_text')
    if input_text is None:
        # 处理找不到输入文本的情况
        return render_template('error.html', message='Input text is missing')
    tokens_with_probs, loss = get_probability_of_text(input_text)
    return render_template('output.html', tokens_with_probs=tokens_with_probs, loss=loss)


@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.form.get('input_text')
    if input_text is None:
        # 处理找不到输入文本的情况
        return render_template('error.html', message='Input text is missing')

    # Fetch the parameters from the form
    temperature = float(request.form.get('temperature'))
    top_p = float(request.form.get('top_p'))
    top_k = int(request.form.get('top_k'))
    num_beams = int(request.form.get('num_beams'))
    max_new_tokens = int(request.form.get('max_new_tokens'))
    min_new_tokens = int(request.form.get('min_new_tokens'))
    repetition_penalty = float(request.form.get('repetition_penalty'))
    do_sample = bool(request.form.get('do_sample'))

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens,  # min_length=min_new_tokens+input_sequence
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )

    tokens_with_probs, loss = decode_and_get_probability(
        input_text, generation_config)
    return render_template('output.html', tokens_with_probs=tokens_with_probs, loss=loss)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
