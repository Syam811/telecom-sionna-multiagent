import os
import numpy as np
import matplotlib.pyplot as plt
from core.sionna_compat import phy_imports


def _qam_constellation(M: int):
    """
    Square M-QAM constellation, Gray-like ordering not required for BER demo.
    Normalized to unit average power.
    Returns complex points array shape [M].
    """
    m_side = int(np.sqrt(M))
    assert m_side * m_side == M, "M must be square for this helper."

    re = np.arange(-(m_side-1), m_side+1, 2)
    im = np.arange(-(m_side-1), m_side+1, 2)
    grid = re[None, :] + 1j * im[:, None]
    pts = grid.flatten()

    # Normalize avg power to 1
    pts = pts / np.sqrt(np.mean(np.abs(pts)**2))
    return pts


def _bits_to_int(bits):
    """bits: [..., k] -> int index"""
    out = np.zeros(bits.shape[:-1], dtype=np.int32)
    for i in range(bits.shape[-1]):
        out = (out << 1) | bits[..., i]
    return out


def _int_to_bits(idx, k):
    """idx: [...] -> bits [..., k]"""
    bits = np.zeros((*idx.shape, k), dtype=np.int32)
    for i in range(k-1, -1, -1):
        bits[..., i] = idx & 1
        idx = idx >> 1
    return bits


def simulate_ber_mimo(
    modulation: str = "64qam",
    snr_db_list=None,
    configs=None,                   # [{"nt":1,"nr":1},{"nt":4,"nr":4}]
    n_bits: int = 30000,            # reduced for CPU safety
    batch_size: int = 200,          # reduced for CPU safety
    out_dir: str = "outputs"
):
    os.makedirs(out_dir, exist_ok=True)
    if snr_db_list is None:
        snr_db_list = [-5, 0, 5, 10, 15]
    if configs is None:
        configs = [{"nt": 1, "nr": 1}, {"nt": 4, "nr": 4}]

    try:
        import tensorflow as tf
        _, Mapper, _, _, FlatFadingChannel, ebnodb2no = phy_imports()
    except Exception as e:
        return {"plots": [], "kpis": {}, "error": f"Sionna/TensorFlow import failed: {e}"}

    mod = modulation.lower()
    if "qam" in mod:
        M = int(mod.replace("qam", ""))
        k = int(np.log2(M))
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=k)
    const_pts = _qam_constellation(M)  # numpy points for hard demap

    all_bers = {}

    for cfg in configs:
        nt, nr = cfg["nt"], cfg["nr"]
        label = f"{nt}x{nr}"
        ch = FlatFadingChannel(num_tx_ant=nt, num_rx_ant=nr, add_awgn=True)

        bers = []

        for snr_db in snr_db_list:
            no = ebnodb2no(snr_db, k, coderate=1.0)

            n_err = 0
            n_tot = 0

            # how many batches needed
            target_bits = n_bits

            while n_tot < target_bits:
                # bits -> symbols
                b = tf.random.uniform([batch_size, k], 0, 2, dtype=tf.int32)
                x = mapper(b)  # [B,1] complex

                # Repeat same symbol across nt TX antennas
                x_mimo = tf.tile(tf.expand_dims(x, axis=2), [1, 1, nt])  # [B,1,nt]

                # Channel
                out = ch(x_mimo, no) if not isinstance(x_mimo, list) else ch([x_mimo, no])
                # Sionna may return y only, or (y,h,...)
                if isinstance(out, tuple):
                    y = out[0]
                    h = out[1] if len(out) > 1 else None
                else:
                    y = out
                    h = None

                # Convert to numpy and REMOVE singleton dims explicitly
                y_np = y.numpy()  # expected [B, 1, nr]
                if h is not None:
                    h_np = h.numpy()  # expected [B, 1, nr, nt]
                else:
                    h_np = np.ones((batch_size, 1, nr, nt), dtype=np.complex64)
                
                # Force exact shapes so numpy can't broadcast to (B,B,...)
                y_np = np.squeeze(y_np, axis=1)          # [B, nr]
                h_np = np.squeeze(h_np, axis=1)          # [B, nr, nt]
                
                # ---- MRC combining (repetition baseline) ----
                # num: sum_{r,t} conj(h[r,t]) * y[r]
                num = np.sum(np.conj(h_np) * y_np[:, :, None], axis=(1, 2))   # [B]
                den = np.sum(np.abs(h_np)**2, axis=(1, 2)) + 1e-9             # [B]
                s_hat = num / den                                             # [B]
                
                # ---- Hard nearest-neighbor demap ----
                d2 = np.abs(s_hat[:, None] - const_pts[None, :])**2           # [B, M]
                sym_idx_hat = np.argmin(d2, axis=1)                           # [B]
                
                # True bits
                b_np = b.numpy().reshape(batch_size, k)                       # [B, k]
                
                # Estimated bits
                b_hat = _int_to_bits(sym_idx_hat, k)                          # [B, k]
                
                # Count errors
                n_err += np.sum(b_hat != b_np)
                n_tot += batch_size * k


            bers.append(n_err / n_tot)

        all_bers[label] = bers

    # Plot
    fig = plt.figure()
    for label, bers in all_bers.items():
        plt.semilogy(snr_db_list, bers, marker="o", label=label)
    plt.title(f"MIMO BER (Hard Demap, Repetition) – {modulation.upper()}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both")
    plt.legend()

    plot_path = os.path.join(out_dir, f"ber_mimo_{mod}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plots": [plot_path],
        "kpis": {
            "configs": configs,
            "snr_db": snr_db_list,
            "ber": all_bers,
            "modulation": modulation,
            "note": "Low-memory hard demap + MRC combining (CPU-friendly)."
        }
    }
