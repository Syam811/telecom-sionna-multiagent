import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_multi_radio_map(
    tx_positions=None,           # list of (x,y,z)
    rx_grid_size=80,
    area_size=(200, 200),
    frequency_hz=3.5e9,
    tx_power_dbm=30.0,
    pathloss_exp=2.2,
    combine_mode="max",          # "max" or "sum"
    out_dir="outputs"
):
    """
    Multi-TX analytical radio map.
    If combine_mode="max": strongest TX dominates (coverage map).
    If "sum": power adds in linear domain.

    Returns JSON with plot path.
    """
    os.makedirs(out_dir, exist_ok=True)

    if tx_positions is None:
        tx_positions = [(0,0,10), (60,0,10), (-60,0,10)]

    w, h = area_size
    xs = np.linspace(-w/2, w/2, rx_grid_size)
    ys = np.linspace(-h/2, h/2, rx_grid_size)

    c = 3e8
    lam = c / frequency_hz
    fspl_const = 20*np.log10(4*np.pi/lam)

    power_maps = []
    for (tx_x, tx_y, tx_z) in tx_positions:
        pmap = np.zeros((rx_grid_size, rx_grid_size))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                d = np.sqrt((x-tx_x)**2 + (y-tx_y)**2 + (tx_z)**2) + 1e-6
                pl_db = fspl_const + 10*pathloss_exp*np.log10(d)
                rx_pwr_dbm = tx_power_dbm - pl_db
                pmap[j, i] = rx_pwr_dbm
        power_maps.append(pmap)

    power_maps = np.stack(power_maps, axis=0)

    if combine_mode == "sum":
        # sum in linear mW then back to dBm
        lin = 10 ** (power_maps/10)
        combined = 10*np.log10(np.sum(lin, axis=0))
    else:
        combined = np.max(power_maps, axis=0)

    fig = plt.figure()
    plt.imshow(combined, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]])
    plt.colorbar(label="Received Power (dBm)")
    for (tx_x, tx_y, _) in tx_positions:
        plt.scatter([tx_x], [tx_y], c="red", marker="^")
    plt.title(f"Multi-TX Radio Map (combine={combine_mode})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plot_path = os.path.join(out_dir, "radio_map_multi_tx.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plots": [plot_path],
        "kpis": {
            "tx_positions": tx_positions,
            "rx_grid_size": rx_grid_size,
            "area_size": area_size,
            "frequency_hz": frequency_hz,
            "combine_mode": combine_mode
        }
    }
