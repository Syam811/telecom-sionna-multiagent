import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_ber_mimo(
    modulation: str = "64qam",
    snr_db_list=None,
    configs=None,                 # e.g., [{"nt":1,"nr":1},{"nt":4,"nr":4}]
    n_bits: int = 150000,
    batch_size: int = 1000,
    out_dir: str = "outputs"
):
    """
    Compares BER for multiple MIMO configs over Rayleigh fading.

    Returns:
      {
        "plots": [<png path>],
        "kpis": {"configs": [...], "snr_db": [...], "ber": { "1x1":[...], "4x4":[...] } }
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    if snr_db_list is None:
        snr_db_list = [-5, 0, 5, 10, 15]
    if configs is None:
        configs = [{"nt": 1, "nr": 1}, {"nt": 4, "nr": 4}]

    try:
        import tensorflow as tf
        from sionna.mapping import Mapper, Demapper, Constellation
        from sionna.channel import FlatFadingChannel
        from sionna.utils import ebnodb2no
    except Exception as e:
        return {"plots": [], "kpis": {}, "error": f"Sionna/TensorFlow not available: {e}"}

    mod = modulation.lower()
    if mod.endswith("qam"):
        m = int(mod.replace("qam", ""))
        constellation = Constellation("qam", m)
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation=constellation)
    demapper = Demapper("app", constellation=constellation)

    n_bits_per_sym = int(np.log2(constellation.num_points))

    all_bers = {}

    for cfg in configs:
        nt, nr = cfg["nt"], cfg["nr"]
        label = f"{nt}x{nr}"

        channel_layer = FlatFadingChannel(
            num_tx_ant=nt,
            num_rx_ant=nr,
            add_awgn=True
        )

        bers = []
        for snr_db in snr_db_list:
            no = ebnodb2no(snr_db, n_bits_per_sym, coderate=1.0)

            n_errors = 0
            n_total = 0

            while n_total < n_bits:
                b = tf.random.uniform([batch_size, n_bits_per_sym], 0, 2, dtype=tf.int32)
                x = mapper(b)

                # Expand to nt antennas (simple repetition baseline)
                x_mimo = tf.tile(tf.expand_dims(x, axis=2), [1, 1, nt])

                y, h = channel_layer([x_mimo, no])
                llr = demapper([y, h, no])

                b_hat = tf.cast(llr > 0, tf.int32)
                n_errors += tf.reduce_sum(tf.cast(tf.not_equal(b, b_hat), tf.int32)).numpy()
                n_total += batch_size * n_bits_per_sym

            bers.append(n_errors / n_total)

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
        "kpis": {"configs": configs, "snr_db": snr_db_list, "ber": all_bers, "modulation": modulation}
    }
