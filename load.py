import torch
import torch.nn as nn
from torch.nn import functional as F
from main import BigramLanguageModel, decode

model = BigramLanguageModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
