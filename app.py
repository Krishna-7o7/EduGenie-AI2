from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load IBM Granite model
MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"
print(f"Loading model: {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

@app.route("/api/answer", methods=["POST"])
def generate_answer():
    data = request.get_json()
    question = data.get("question", "")
    contexts = data.get("contexts", [])

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Combine contexts (top 3 retrieved chunks)
    context_text = "\n\n".join(contexts[:3]) if contexts else "No context available."

    # Prompt formatting for the Granite model
    messages = [
        {"role": "system", "content": "You are StudyMate, an AI that answers questions based on provided study materials."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.6)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return jsonify({"answer": result.strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
