import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, json


def load_gsm8k_with_answers(file_path, num_samples=50):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q = data.get('question', '')
            a_str = data.get('answer', '')
            if q and "####" in a_str:
                true_ans_str = a_str.split("####")[1].strip().replace(",", "")
                dataset.append({"raw_prompt": q, "true_answer": true_ans_str})
            if len(dataset) >= num_samples: break
    return dataset


def extract_last_number(text):
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    return numbers[-1] if numbers else None


def run_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_or_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    gsm8k_path = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(device).eval()
    gsm8k_data = load_gsm8k_with_answers(gsm8k_path, 15)  # 跑15个够看就行

    wrong_count = 0

    print("🚀 开始抓取 LLaMA-3 的真实案发现场...")
    for i, item in enumerate(gsm8k_data):
        messages = [
            {"role": "system",
             "content": "You are a math expert. Solve the problem step by step. Conclude your answer by saying 'The final answer is [number]'."},
            {"role": "user", "content": item["raw_prompt"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,  # 🎯 长度暴增到 512
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        pred_ans = extract_last_number(generated_text)

        if pred_ans != item["true_answer"]:
            wrong_count += 1
            print("\n" + "=" * 60)
            print(f"❌ 错题 #{i + 1}")
            print(f"【标准答案】: {item['true_answer']}")
            print(f"【我们提取的】: {pred_ans}")
            print(f"【LLaMA-3 真实输出】:\n{generated_text}")
            print("=" * 60)

            if wrong_count >= 3:
                break


if __name__ == "__main__":
    run_debug()