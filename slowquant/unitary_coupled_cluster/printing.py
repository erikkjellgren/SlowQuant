import time

KEY_WIDTH = 34


def key_value(key: str, value) -> str:
    return f"   {key:<{KEY_WIDTH}}{value}"


class IterationTable:
    """Fixed-width iteration table: title, settings, aligned columns (name, width, format), summary."""

    def __init__(self, title: str, columns: list[tuple[str, int, str]], info=(), show: bool = True):
        self.cols = columns
        self.show = show
        self.width = max(1 + sum(w for _, w, _ in columns) + 2 * (len(columns) - 1), 72)
        self.t = time.time()
        if show:
            print()
            print("=" * self.width)
            print(f" {title}")
            for k, v in info:
                print(key_value(k, v))
            print("-" * self.width)
            print(" " + "  ".join(f"{n:{'<' if f == '' else '>'}{w}}" for n, w, f in columns))
            print("-" * self.width, flush=True)

    def lap(self) -> float:
        t, self.t = self.t, time.time()
        return self.t - t

    def row(self, *values) -> None:
        if not self.show:
            return
        cells = [
            f"{'-' if v is None else v:{'<' if f == '' else '>'}{w}}" if v is None or isinstance(v, str) else f"{v:>{w}{f}}"
            for v, (_, w, f) in zip(values, self.cols)
        ]
        print(" " + "  ".join(cells), flush=True)

    def close(self, status: str, info=()) -> None:
        if not self.show:
            return
        if status or info:
            print("-" * self.width)
        if status:
            print(f" {status}")
        for k, v in info:
            print(key_value(k, v))
        print("=" * self.width, flush=True)


HARTREE_EV = 27.211386245988


def value_table(title: str, columns: list[tuple[str, int, str]], rows, status: str = "", info=()) -> None:
    """Print a finished table (no timing)."""
    tab = IterationTable(title, columns, info)
    for r in rows:
        tab.row(*r)
    tab.close(status)


def roots_table(title: str, energies, extra=()) -> None:
    """Excitation energies in Eh and eV, plus extra columns (name, values, format)."""
    cols = [("Root", 4, "d"), ("omega [Eh]", 14, ".8f"), ("omega [eV]", 12, ".6f")] + [(n, max(len(n), 12), f) for n, _, f in extra]
    rows = [(i + 1, w, w * HARTREE_EV, *[v[i] for _, v, _ in extra]) for i, w in enumerate(energies)]
    value_table(title, cols, rows, f"{len(energies)} roots")


def timing_line(label: str, seconds: float) -> None:
    print(key_value(f"{label} [s]", f"{seconds:20.1f}"), flush=True)


def info_block(title: str, items, width: int = 86) -> None:
    """Key-value block: (key, value) pairs."""
    print()
    print("=" * width)
    print(f" {title}")
    print("-" * width)
    for k, v in items:
        print(key_value(k, v))
    print("=" * width, flush=True)
