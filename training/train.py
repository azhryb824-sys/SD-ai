import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model import GreetingLanguageModel


def main():
    random.seed(42)
    torch.manual_seed(42)

    data_path = ROOT / "data" / "dataset.json"
    with data_path.open(encoding="utf-8") as file:
        examples = json.load(file)

    sequences = [f"المستخدم: {item['prompt']}\nالمساعد: {item['response']}§" for item in examples]
    chars = sorted(set("".join(sequences)))
    char_to_id = {char: index for index, char in enumerate(chars)}

    encoded = [
        torch.tensor([char_to_id[char] for char in sequence], dtype=torch.long)
        for sequence in sequences
    ]

    model = GreetingLanguageModel(len(chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = 220
    for epoch in range(1, epochs + 1):
        random.shuffle(encoded)
        total_loss = 0.0
        for sequence in encoded:
            inputs = sequence[:-1].unsqueeze(0)
            targets = sequence[1:].unsqueeze(0)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, len(chars)), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch={epoch:03d} loss={total_loss / len(encoded):.4f}", flush=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "char_to_id": char_to_id,
        "config": {"vocab_size": len(chars), "embedding_dim": 48, "hidden_dim": 96},
    }
    output_path = ROOT / "models" / "greeting_model.pt"
    torch.save(checkpoint, output_path)
    print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
