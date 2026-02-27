"""
BS-RoFormer ONNX Conversion Script
Downloads the karaoke BS-RoFormer checkpoint and exports to ONNX format.
Uses Conv1D-based STFT/ISTFT to avoid complex tensor operations that ONNX
does not support.

Usage:
    python scripts/convert_bsroformer.py

Output:
    src-tauri/models/bsroformer.onnx
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

CHECKPOINT_URL = "https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.ckpt"
CONFIG_URL = "https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.yaml"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "src-tauri", "models")
CACHE_DIR = os.path.join(MODELS_DIR, ".cache")


def download_file(url, dest):
    import urllib.request
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  Done ({size_mb:.1f} MB)")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)
    return config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conv1D STFT / ISTFT  (fully real-valued, ONNX safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Conv1DSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq = n_fft // 2 + 1

        window = torch.hann_window(win_length)
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window = F.pad(window, (left, n_fft - win_length - left))

        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(self.n_freq, dtype=torch.float32)
        angles = -2.0 * math.pi * k.unsqueeze(1) * n.unsqueeze(0) / n_fft

        real_k = torch.cos(angles) * window.unsqueeze(0)
        imag_k = torch.sin(angles) * window.unsqueeze(0)
        kernel = torch.cat([real_k, imag_k], dim=0).unsqueeze(1)   # [2*n_freq, 1, n_fft]
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        """x: [B, samples] -> [B, n_freq, frames, 2]"""
        pad = self.n_fft // 2
        x = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect')
        out = F.conv1d(x, self.kernel, stride=self.hop_length)
        real = out[:, :self.n_freq, :]
        imag = out[:, self.n_freq:, :]
        return torch.stack([real, imag], dim=-1)


class Conv1DISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq = n_fft // 2 + 1

        window = torch.hann_window(win_length)
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window = F.pad(window, (left, n_fft - win_length - left))

        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(self.n_freq, dtype=torch.float32)
        angles = 2.0 * math.pi * k.unsqueeze(1) * n.unsqueeze(0) / n_fft

        weights = torch.ones(self.n_freq) * 2.0 / n_fft
        weights[0] = 1.0 / n_fft
        if n_fft % 2 == 0:
            weights[-1] = 1.0 / n_fft

        real_k = torch.cos(angles) * weights.unsqueeze(1)
        imag_k = -torch.sin(angles) * weights.unsqueeze(1)

        # synthesis kernel: [2*n_freq, 1, n_fft]
        kernel = torch.cat([real_k, imag_k], dim=0).unsqueeze(1)
        self.register_buffer('kernel', kernel)

        # squared window for OLA normalisation
        self.register_buffer('window_sq', window * window)

    def forward(self, stft_repr, orig_len):
        """
        stft_repr: [B, n_freq, frames, 2]
        orig_len:  int (desired output length)
        -> [B, orig_len]
        """
        real = stft_repr[..., 0]
        imag = stft_repr[..., 1]
        x = torch.cat([real, imag], dim=1)      # [B, 2*n_freq, frames]

        audio = F.conv_transpose1d(x, self.kernel, stride=self.hop_length)  # [B, 1, L]
        audio = audio.squeeze(1)

        # OLA window envelope (pre-computed for fixed frame count is OK at trace time)
        n_frames = stft_repr.shape[2]
        L = audio.shape[1]
        envelope = torch.zeros(L, device=audio.device)
        for i in range(n_frames):
            s = i * self.hop_length
            e = s + self.n_fft
            if e <= L:
                envelope[s:e] = envelope[s:e] + self.window_sq

        envelope = torch.clamp(envelope, min=1e-8)
        audio = audio / envelope.unsqueeze(0)

        pad = self.n_fft // 2
        audio = audio[:, pad:pad + orig_len]
        return audio


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONNX-compatible wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BSRoformerOnnxWrapper(nn.Module):
    """
    Input : [1, 1, samples] mono PCM 44100 Hz
    Output: [1, 2, samples] (vocal, accompaniment)
    """
    def __init__(self, model, stft_kwargs, audio_channels=2):
        super().__init__()
        # keep a reference to the *original* model so its parameters are included
        self.inner = model
        self.audio_channels = audio_channels
        self.num_stems = model.num_stems

        n_fft = stft_kwargs['n_fft']
        hop   = stft_kwargs['hop_length']
        win   = stft_kwargs['win_length']
        self.stft_mod  = Conv1DSTFT(n_fft, hop, win)
        self.istft_mod = Conv1DISTFT(n_fft, hop, win)

    def forward(self, x):
        from einops import rearrange, pack, unpack

        batch, _ch, samples = x.shape
        stereo = x.expand(batch, 2, samples)
        raw = stereo.reshape(batch * 2, samples)

        # STFT  -> [B*2, freq, frames, 2]
        stft_repr = self.stft_mod(raw)
        n_freq, n_frames = stft_repr.shape[1], stft_repr.shape[2]
        stft_repr = stft_repr.reshape(batch, 2, n_freq, n_frames, 2)  # [b,s,f,t,c]

        # merge stereo into freq  "b s f t c -> b (f s) t c"
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(batch, n_freq * 2, n_frames, 2)

        # "b f t c -> b t (f c)"
        x_proc = stft_repr.permute(0, 2, 1, 3).reshape(batch, n_frames, -1)

        # band split
        x_proc = self.inner.band_split(x_proc)

        # transformer layers
        for tb in self.inner.layers:
            if len(tb) == 3:
                linear_t, time_t, freq_t = tb
                x_proc, ft_ps = pack([x_proc], "b * d")
                x_proc = linear_t(x_proc)
                (x_proc,) = unpack(x_proc, ft_ps, "b * d")
            else:
                time_t, freq_t = tb

            x_proc = rearrange(x_proc, "b t f d -> b f t d")
            x_proc, ps = pack([x_proc], "* t d")
            x_proc = time_t(x_proc)
            (x_proc,) = unpack(x_proc, ps, "* t d")
            x_proc = rearrange(x_proc, "b f t d -> b t f d")
            x_proc, ps = pack([x_proc], "* f d")
            x_proc = freq_t(x_proc)
            (x_proc,) = unpack(x_proc, ps, "* f d")

        x_proc = self.inner.final_norm(x_proc)

        # mask
        mask = torch.stack([fn(x_proc) for fn in self.inner.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        # complex mul  (a+bi)(c+di) = (ac-bd)+(ad+bc)i
        stft_e = stft_repr.unsqueeze(1)
        a, b = stft_e[..., 0], stft_e[..., 1]
        c, d = mask[..., 0],   mask[..., 1]
        res = torch.stack([a*c - b*d, a*d + b*c], dim=-1)   # [b,n,f*s,t,2]

        # reshape for ISTFT  "b n (f s) t c -> (b n s) f t c"
        s = self.audio_channels
        res = res.reshape(batch, self.num_stems, n_freq, s, n_frames, 2)
        res = res.permute(0, 1, 3, 2, 4, 5).reshape(batch * self.num_stems * s, n_freq, n_frames, 2)

        recon = self.istft_mod(res, samples)
        recon = recon.reshape(batch, self.num_stems, s, -1)

        if self.num_stems == 1:
            recon = recon.squeeze(1)                     # [b, s, t]

        vocals_mono = recon.mean(dim=1, keepdim=True)    # [b, 1, t]
        acc_mono = x[:, :, :vocals_mono.shape[-1]] - vocals_mono
        return torch.cat([vocals_mono, acc_mono], dim=1) # [b, 2, t]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_model(config):
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer

    m = config['model']
    args = dict(
        dim=m['dim'], depth=m['depth'],
        stereo=m.get('stereo', False),
        num_stems=m.get('num_stems', 2),
        time_transformer_depth=m.get('time_transformer_depth', 2),
        freq_transformer_depth=m.get('freq_transformer_depth', 2),
        freqs_per_bands=tuple(m['freqs_per_bands']),
        dim_head=m.get('dim_head', 64),
        heads=m.get('heads', 8),
        attn_dropout=m.get('attn_dropout', 0.0),
        ff_dropout=m.get('ff_dropout', 0.0),
        flash_attn=False,
        mlp_expansion_factor=m.get('mlp_expansion_factor', 4),
        use_torch_checkpoint=False,
        skip_connection=m.get('skip_connection', False),
    )
    for k in ('stft_n_fft', 'stft_hop_length', 'stft_win_length'):
        if k in m:
            args[k] = m[k]

    print(f"  dim={args['dim']}, depth={args['depth']}, stereo={args['stereo']}, stems={args['num_stems']}")
    return BSRoformer(**args)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    output_path = os.path.join(MODELS_DIR, "bsroformer.onnx")

    # 1) download
    print("[1/4] Downloading checkpoint and config...")
    ckpt_path   = os.path.join(CACHE_DIR, "karaoke_bs_roformer_anvuew.ckpt")
    config_path = os.path.join(CACHE_DIR, "karaoke_bs_roformer_anvuew.yaml")
    download_file(CONFIG_URL, config_path)
    download_file(CHECKPOINT_URL, ckpt_path)

    # 2) config
    print("\n[2/4] Loading configuration...")
    config = load_config(config_path)
    stft_kwargs = dict(
        n_fft=config['model']['stft_n_fft'],
        hop_length=config['model']['stft_hop_length'],
        win_length=config['model']['stft_win_length'],
    )
    print(f"  STFT: {stft_kwargs}")

    # 3) model + weights
    print("\n[3/4] Creating model and loading weights...")
    model = create_model(config)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    elif isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd)
    model.eval()
    print("  Weights loaded.")

    # quick sanity check
    with torch.no_grad():
        t = torch.randn(1, 2, 131072)
        o = model(t)
        print(f"  Original: {t.shape} -> {o.shape}")

    # 4) wrap & export
    print("\n[4/4] Wrapping and exporting to ONNX...")
    wrapped = BSRoformerOnnxWrapper(model, stft_kwargs, audio_channels=2)
    wrapped.eval()

    with torch.no_grad():
        test = torch.randn(1, 1, 131072)
        out  = wrapped(test)
        print(f"  Wrapper: {test.shape} -> {out.shape}")

    dummy = torch.randn(1, 1, 131072)
    print("  torch.onnx.export ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped, dummy, output_path,
            input_names=["mix"],
            output_names=["separated"],
            dynamic_axes={"mix": {2: "samples"}, "separated": {2: "samples"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    mb = os.path.getsize(output_path) / (1024*1024)
    print(f"\n  Saved {output_path}  ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
