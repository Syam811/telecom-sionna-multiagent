import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_radio_map(
    tx_pos=(0, 0, 10),
    rx_grid_size=80,
    area_size=(200, 200),      # meters
    frequency_hz=3.5e9,
    tx_power_dbm=30.0,
    pathloss_exp=2.2,
    out_dir="outputs"
):
    """
    Simple analytical radio map (pathloss-based) if ray tracing not available.
    If you already have Sionna RT pipeline, replace internals.

    Returns:
      {
        "plots": [<png path>],
        "kpis": {"tx_pos":..., "frequency_hz":..., "rx_grid_size":...}
      }
    """
    os.makedirs(out_dir, exist_ok=True)

    tx_x, tx_y, tx_z = tx_pos
    w, h = area_size

    xs = np.linspace(-w/2, w/2, rx_grid_size)
    ys = np.linspace(-h/2, h/2, rx_grid_size)

    power_map = np.zeros((rx_grid_size, rx_grid_size))

    # Free-space + pathloss exponent approximation
    c = 3e8
    lam = c / frequency_hz
    fspl_const = 20*np.log10(4*np.pi/lam)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            d = np.sqrt((x-tx_x)**2 + (y-tx_y)**2 + (tx_z)**2) + 1e-6
            pl_db = fspl_const + 10*pathloss_exp*np.log10(d)
            rx_pwr_dbm = tx_power_dbm - pl_db
            power_map[j, i] = rx_pwr_dbm

    fig = plt.figure()
    plt.imshow(power_map, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]])
    plt.colorbar(label="Received Power (dBm)")
    plt.scatter([tx_x], [tx_y], c="red", marker="^", label="TX")
    plt.title("Radio Map (Analytical Pathloss)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()

    plot_path = os.path.join(out_dir, "radio_map_single_tx.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plots": [plot_path],
        "kpis": {
            "tx_pos": tx_pos,
            "rx_grid_size": rx_grid_size,
            "area_size": area_size,
            "frequency_hz": frequency_hz
        }
    }
