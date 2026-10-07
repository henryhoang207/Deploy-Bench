import torch, transformers
from transformers import Wav2Vec2Model
P = "/mnt/data8tb/henry/deploybench/vendor/wav2vec2-xls-r-2b"
print("transformers", transformers.__version__, "torch", torch.__version__)
sd = torch.load(P + "/pytorch_model.bin", map_location="cpu")
ks = [k for k in sd if "pos_conv" in k]
print("checkpoint pos_conv keys:", ks)
g, v = sd["wav2vec2.encoder.pos_conv_embed.conv.weight_g"], sd["wav2vec2.encoder.pos_conv_embed.conv.weight_v"]
w_ckpt = torch._weight_norm(v, g, 2)
for trial in range(2):
    torch.manual_seed(trial)
    m = Wav2Vec2Model.from_pretrained(P)
    conv = m.encoder.pos_conv_embed.conv
    w = conv.weight.detach()
    print(f"trial {trial}: max|w_model - w_ckpt| = {float((w - w_ckpt).abs().max()):.3e}; param names:",
          [n for n, _ in conv.named_parameters()])
