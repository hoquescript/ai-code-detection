import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_prompt(code: str) -> list[dict[str, str]]:
    prompt = f"""You are given a piece of code written by a human. Your task is to produce an alternative version of this code.

Guidelines:
- Use the same programming language.
- Preserve the original functionality and behavior exactly.
- You may refactor structure, rename variables, adjust formatting, or use equivalent constructs.
- Do not intentionally simplify or over-complicate the code.
- Do not add comments or stylistic markers that reveal authorship.

STRICT OUTPUT FORMAT:
- Output ONLY raw code.
- Do NOT include markdown formatting.
- Do NOT include triple backticks (```).
- Do NOT include language labels like "java", "python", etc.
- Do NOT include any explanation or extra text.
- The output must start directly with code and end with code.

Original code:
{code}
"""
    messages = [
        {
            "role": "system",
            "content": "You output only raw code. Never include markdown, backticks, or explanations.",
        },
        {"role": "user", "content": prompt},
    ]
    return messages


def generate_ai_pair(messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    df = pd.read_json("data/contrastive-aidev/java_codegptsensor.jsonl", lines=True)
    sample_df = df.sample(n=5)
    sample_df = sample_df[sample_df["code"].str.len() < 20000]
    human_df = sample_df[sample_df["label"] == 1]
    for index, row in human_df.iterrows():
        code_human = row["code"]
        prompt = get_prompt(code_human)
        code_ai = generate_ai_pair(prompt)
        code_ai = code_ai.replace("```java", "").replace("```", "").strip()
        human_df.loc[index, "contrast"] = code_ai
    human_df.to_json(
        "data/contrastive-aidev/java_codegptsensor_paired.jsonl",
        orient="records",
        lines=True,
        mode="w",
    )
