import torch
from fairseq_signals_backbone.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel, Wav2Vec2CMSCConfig


encoder_embed_dim = 768
encoder_attention_heads = 12
encoder_layers = 8
encoder_ffn_embed_dim = 3072

cfg = Wav2Vec2CMSCConfig(
    apply_mask = True,
    mask_prob = 0.65,
    quantize_targets = True,
    final_dim = 256,
    dropout_input = 0.1,
    dropout_features = 0.1,
    feature_grad_mult = 0.1,
    encoder_embed_dim = encoder_embed_dim,
    encoder_attention_heads = encoder_attention_heads,
    in_d = 12,
    encoder_layers = encoder_layers,
    encoder_ffn_embed_dim = encoder_ffn_embed_dim
)
ecg_encoder = Wav2Vec2CMSCModel(cfg)
print(ecg_encoder)

ecg = torch.randn(1, 12, 5000)
out = ecg_encoder(source=ecg, mask=False, features_only=True)
print(out)