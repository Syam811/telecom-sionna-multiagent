import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_constellation(
    modulation: str = "16qam",
    snr_db: float = 15.0,
    n_symbols: int = 2000,
    out_dir: str = "outputs"
):
    # Ensure folder exists
    os.makedirs(out_dir, exist_ok=True)

    # Safe import
    try:
        import tensorflow as tf
        from sionna.mapping import Mapper, Constellation
        from sionna.channel import AWGN
    except Exception as e:
        return {
            "plots": [],
            "kpis": {},
            "error": f"Sionna/TensorFlow not available: {e}"
        }

    # Map modulation string → Sionna constellation
    mod = modulation.lower()

    if mod == "qpsk":
        constellation = Constellation("qam", 4)
    elif "qam" in mod:
        m = int(mod.replace("qam", ""))
        constellation = Constellation("qam", m)
    else:
        return {"plots": [], "kpis": {}, "error": f"Unknown modulation: {modulation}"}

    mapper = Mapper(constellation)
    awgn = AWGN()

    n_bits_per_sym = int(np.log2(constellation.num_points))

    # Generate random symbols
    bits = tf.random.uniform(
        [n_symbols, n_bits_per_sym],
        minval=0, maxval=2, dtype=tf.int32
    )
    x = mapper(bits)

    # Add channel noise
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0 / snr_linear
    y = awgn([x, tf.constant(noise_var, tf.float32)])

    # Convert to numpy complex values
    y_np = y.numpy().reshape(-1)

    # Plotting
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(np.real(y_np), np.imag(y_np), s=6, alpha=0.6)
    plt.title(f"Constellation: {modulation.upper()} @ {snr_db} dB")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)

    # Save figure
    plot_path = os.path.join(out_dir, f"constellation_{mod}_{snr_db}db.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    # Return successful result
    return {
        "plots": [plot_path],
        "kpis": {
            "modulation": modulation,
            "snr_db": snr_db,
            "n_symbols": n_symbols
        }
    }
