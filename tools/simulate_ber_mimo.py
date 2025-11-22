import os
import numpy as np
import matplotlib.pyplot as plt
from core.sionna_compat import phy_imports

def simulate_ber_mimo(
    modulation: str = "64qam",
    snr_db_list=None,
    configs=None,                   # [{"nt":1,"nr":1},{"nt":4,"nr":4}]
    n_bits: int = 150000,
    batch_size: int = 1000,
    out_dir: str = "outputs"
):
    os.makedirs(out_dir, exist_ok=True)
    if snr_db_list is None:
        snr_db_list = [-5, 0, 5, 10, 15]
    if configs is None:
        configs = [{"nt": 1, "nr": 1}, {"nt": 4, "nr": 4}]

    try:
        import tensorflow as tf
        _, Mapper, Demapper, _, FlatFadingChannel, ebnodb2no = phy_imports()
    except Exception as e:
        return {"plots": [], "kpis": {}, "error": f"Sionna/TensorFlow import failed: {e}"}

    mod = modulation.lower()

    if "qam" in mod:
        M = int(mod.replace("qam", ""))
        k = int(np.log2(M))
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=k)
    demapper = Demapper("app", constellation_type="qam", num_bits_per_symbol=k)

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

            while n_tot < n_bits:
                b = tf.random.uniform([batch_size, k], 0, 2, dtype=tf.int32)
                x = mapper(b)  # shape: [B, 1] complex symbols

                # Simple repetition baseline across nt antennas
                x_mimo = tf.tile(tf.expand_dims(x, axis=2), [1, 1, nt])

                try:
                    y, h = ch(x_mimo, no)
                except TypeError:
                    y, h = ch([x_mimo, no])

                llr = demapper(y, h, no) if hasattr(demapper, "__call__") else demapper([y, h, no])

                b_hat = tf.cast(llr > 0, tf.int32)
                n_err += tf.reduce_sum(tf.cast(tf.not_equal(b, b_hat), tf.int32)).numpy()
                n_tot += batch_size * k

            bers.append(n_err / n_tot)

        all_bers[label] = bers

    # Plot
    fig = plt.figure()
    for label, bers in all_bers.items():
        plt.semilogy(snr_db_list, bers, marker="o", label=label)

    plt.title(f"MIMO BER Comparison ({modulation.upper()} - Rayleigh)")
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
            "modulation": modulation
        }
    }
