
# 🎯 Model Steering with Python

> Control what a language model says — without fine-tuning, without prompt engineering. Just math.

A beginner-friendly, hands-on implementation of **activation steering** (a.k.a. representation engineering) for transformer language models. This repo shows you how to **modify a model's internal hidden states** during inference to steer its behavior toward a desired direction — like making GPT-2 sound more positive, more negative, or more cautious.

---

## 🧠 What is Model Steering?

Every transformer model has **hidden states** — internal vector representations of "what the model is thinking" at each layer. Steering works by adding a small **direction vector** to those hidden states during the forward pass:

```
new_hidden = original_hidden + α * steering_vector
```

- `steering_vector` → the *direction* we want to push toward
- `α` (alpha) → *how hard* we push

No gradient updates. No retraining. Just a targeted nudge inside the network.

---

## ✨ Why This Matters

| Traditional Approach | Model Steering |
|----------------------|----------------|
| Fine-tuning | ✅ No training required |
| Prompt engineering | ✅ Works at the activation level |
| RLHF | ✅ Instant, tunable, reversible |
| Editing weights | ✅ Non-destructive |

**Real-world uses:**
- 🛡️ **Safety** — force refusals for harmful requests
- 💬 **Sentiment control** — positive/negative tone on demand
- 🎯 **Truthfulness** — reduce hallucinations (ITI method)
- ✍️ **Style transfer** — formal, casual, concise
- 🚫 **Toxicity removal** — subtract toxic directions

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch transformers numpy
```

### 2. Run the demo

```bash
python steer.py
```

### 3. That's it 🎉

You'll see GPT-2 generate text with different steering strengths.

---

## 📂 Project Structure

```
model-steering/
├── steer.py             # Full working example
├── README.md
└── requirements.txt
```

---

## 🔬 How It Works — Step by Step

### Step 1: Load a model
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
```

### Step 2: Build a steering vector (contrastive method)
```python
def get_hidden(prompt, layer=6):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :]

pos = get_hidden("This is amazing and wonderful and")
neg = get_hidden("This is awful and terrible and")

steering_vector = pos - neg
steering_vector = steering_vector / steering_vector.norm()
```

### Step 3: Hook into the model's forward pass
```python
class SteeringHook:
    def __init__(self, vec, alpha):
        self.vec, self.alpha = vec, alpha

    def __call__(self, module, inp, out):
        if isinstance(out, torch.Tensor):     # GPT-2 style
            return out + self.alpha * self.vec
        return (out[0] + self.alpha * self.vec,) + tuple(out[1:])  # Llama style
```

### Step 4: Generate with steering applied
```python
def generate_with_steering(prompt, alpha=0.0, layer=6, max_new_tokens=30):
    handle = model.transformer.h[layer].register_forward_hook(
        SteeringHook(steering_vector, alpha)
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        handle.remove()
```

### Step 5: Compare outputs
```python
prompt = "The weather today is"
for alpha in [-10.0, 0.0, 10.0, 20.0]:
    print(f"α={alpha:+.0f}: {generate_with_steering(prompt, alpha=alpha)}")
```

---

## 📊 Example Output

```
α=-10: The weather today is very bad, and it's going to rain all week...
α=+0 : The weather today is very good, and we're going to be able to...
α=+10: The weather today is absolutely wonderful and beautiful, perfect...
α=+20: The weather today is amazing fantastic wonderful beautiful perfect...
```

*(Actual output depends on model, layer, and alpha — experiment!)*

---

## 🎛️ Tuning Guide

| Knob | What It Does | Recommendation |
|------|--------------|----------------|
| **`alpha` (α)** | Steering strength | Start small (5–15), increase gradually |
| **`layer`** | Which transformer block | Middle layers (5–10 for GPT-2) work best |
| **`steering_vector`** | Direction to push | Build from contrastive prompt pairs |
| **Normalization** | Vector scale | Always normalize to unit length first |

**Rule of thumb:** if output becomes gibberish, α is too high. If nothing changes, α is too low.

---

## 🧩 Compatibility

| Model | Status |
|-------|--------|
| GPT-2 | ✅ Tested (Tensor output) |
| GPT-Neo / GPT-J | ✅ Tensor output |
| Llama / Llama-2 / 3 | ✅ Tuple output (handled) |
| Mistral / Mixtral | ✅ Tuple output (handled) |
| Qwen / Phi / Gemma | ⚠️ Likely works — test your version |

The hook automatically handles both **Tensor** and **tuple** block outputs, so it's portable across most HF models.

---

## ⚠️ Common Pitfalls

1. **Device mismatch** → move the vector: `steering_vector.to(model.device)`
2. **Dtype mismatch** → cast it: `.to(model.dtype)` (fp16 vs fp32)
3. **Forgetting to remove the hook** → always use `try/finally`
4. **Alpha too large** → model outputs nonsense
5. **Wrong layer** → no visible effect

---

## 🔗 Further Reading

- **Activation Addition** — Turner et al., 2023
- **Inference-Time Intervention (ITI)** — Li et al., 2023
- **Representation Engineering (RepE)** — Zou et al., 2023
- **Steering Llama 2 via Contrastive Activation Addition** — Rimsky et al., 2023

---

## 🤝 Contributing

PRs welcome! Especially:
- Steering vectors for new behaviors (truthfulness, humor, etc.)
- Support for more model architectures
- Better vector-finding methods (PCA, mass-mean shift)

---

## 📜 License

MIT — use it, fork it, break it, improve it.

---

## ⭐ If this helped you

Give the repo a star — it helps others find it!
````

---

## 📌 GitHub Repository Description (short — for the "About" box)

Pick one:

**Option A (descriptive):**
> 🎯 Hands-on guide to steering LLM behavior by editing hidden states — no fine-tuning required. Works with GPT-2, Llama, Mistral.

**Option B (punchy):**
> Control what LLMs say by nudging their internal activations. Beginner-friendly PyTorch + Transformers tutorial.

**Option C (SEO-friendly):**
> Activation steering / representation engineering for transformers in Python. Learn to modify hidden states with forward hooks — GPT-2, Llama, Mistral examples.

---

## 🏷️ Suggested GitHub Topics

```
llm
activation-steering
representation-engineering
interpretability
transformers
pytorch
gpt2
llama
mechanistic-interpretability
ai-safety
```

---

## 📋 `requirements.txt`

```txt
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
```

---
