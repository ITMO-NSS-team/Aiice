from typing import Sequence

import matplotlib.pyplot as plt


def plot_history(history: Sequence[float], save_path: str, show: bool = True) -> None:
    plt.plot(list(range(len(history))), history)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"History per epoch ({save_path})")
    plt.savefig(f"{save_path}")

    if show:
        plt.show()
    plt.close()
