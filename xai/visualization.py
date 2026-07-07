from pathlib import Path
import matplotlib.pyplot as plt


def plot_vector(values, out_path, title='importance'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel('index')
    plt.ylabel('importance')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
