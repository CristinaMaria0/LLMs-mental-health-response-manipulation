"""Shared plotting helpers."""


def annotated_heatmap(ax, pivot, title, fig, vmax=1.0, fmt="{:.2f}", cmap="Reds", cbar_label=None):
    """Draw an annotated heatmap of a pandas pivot onto `ax`."""
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, "-" if (v != v) else fmt.format(v), ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label or "")
    return im
