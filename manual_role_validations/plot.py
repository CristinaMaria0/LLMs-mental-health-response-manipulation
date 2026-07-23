from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def generate_manual_behavior_plots(
    llama_file,
    gemma_file,
    qwen_file,
    output_dir="outputs/summary",
):
    """Generate individual and combined behaviour-mix plots from manual annotations."""

    files = {
        "Llama 3.1 8B": Path(llama_file),
        "Gemma 3 4B": Path(gemma_file),
        "Qwen3 4B": Path(qwen_file),
    }

    roles = ["victim", "perpetrator"]
    intents = ["help_seeking", "validation", "how_to"]

    cells = [
        f"{role} / {intent}"
        for role in roles
        for intent in intents
    ]

    behaviors = [
        "empathetic_support",
        "safe_redirect",
        "challenge_repair",
        "sycophantic_validation",
        "enabling_howto",
        "dismissive",
        "other",
    ]

    behavior_colors = {
        "empathetic_support": "#2ecc71",
        "safe_redirect": "#27ae60",
        "challenge_repair": "#16a085",
        "sycophantic_validation": "#e67e22",
        "enabling_howto": "#e74c3c",
        "dismissive": "#8e44ad",
        "other": "#95a5a6",
    }

    behavior_labels = {
        "empathetic_support": "Empathetic support",
        "safe_redirect": "Safe redirect",
        "challenge_repair": "Challenge and repair",
        "sycophantic_validation": "Sycophantic validation",
        "enabling_howto": "Enabling how-to",
        "dismissive": "Dismissive",
        "other": "Other",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compositions = {}

    # Read and prepare each model's manual annotations.
    for model_name, file_path in files.items():
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.astype(str).str.strip()

        # The files use both British and American spellings.
        if "h_behaviour" in df.columns:
            behavior_column = "h_behaviour"
        elif "h_behavior" in df.columns:
            behavior_column = "h_behavior"
        else:
            raise KeyError(
                f"No manual behaviour column found in {file_path}. "
                "Expected 'h_behaviour' or 'h_behavior'."
            )

        required_columns = {"role", "intent", behavior_column}
        missing = required_columns.difference(df.columns)

        if missing:
            raise KeyError(
                f"Missing columns in {file_path}: {sorted(missing)}"
            )

        # Use only manually annotated behaviour labels.
        manual_df = df.dropna(subset=[behavior_column]).copy()

        manual_df["role"] = (
            manual_df["role"].astype(str).str.strip().str.lower()
        )
        manual_df["intent"] = (
            manual_df["intent"].astype(str).str.strip().str.lower()
        )
        manual_df["manual_behavior"] = (
            manual_df[behavior_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Construct the cell consistently rather than relying on an
        # existing cell column.
        manual_df["cell"] = (
            manual_df["role"] + " / " + manual_df["intent"]
        )

        unknown = sorted(
            set(manual_df["manual_behavior"]) - set(behaviors)
        )

        if unknown:
            raise ValueError(
                f"Unknown manual behaviour labels in {file_path}: "
                f"{unknown}"
            )

        counts = (
            manual_df
            .groupby(["cell", "manual_behavior"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=cells, columns=behaviors, fill_value=0)
        )

        # Convert counts to proportions within each role-intent cell.
        totals = counts.sum(axis=1)
        proportions = (
            counts
            .div(totals.replace(0, np.nan), axis=0)
            .fillna(0)
        )

        compositions[model_name] = proportions

        print(
            f"{model_name}: {len(manual_df)} manually annotated "
            f"responses loaded from {file_path}"
        )

    def draw_model(ax, model_name, composition):
        """Draw one stacked behaviour-mix plot."""

        x = np.arange(len(cells))
        bottom = np.zeros(len(cells))

        for behavior in behaviors:
            values = composition[behavior].to_numpy()

            if values.sum() == 0:
                continue

            ax.bar(
                x,
                values,
                bottom=bottom,
                color=behavior_colors[behavior],
                width=0.82,
                edgecolor="white",
                linewidth=0.5,
                label=behavior_labels[behavior],
            )

            bottom += values

        labels = [
            cell.replace(" / ", "\n").replace("_", " ")
            for cell in cells
        ]

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion of responses")
        ax.set_title(model_name, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    # Generate one plot for each model.
    for model_name, composition in compositions.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        draw_model(ax, model_name, composition)

        handles, labels = ax.get_legend_handles_labels()

        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=7,
                ncol=2,
                loc="lower center",
                frameon=True,
            )

        fig.tight_layout()

        filename = (
            model_name.lower()
            .replace(" ", "_")
            .replace(".", "")
            + "_manual_behavior_mix.png"
        )

        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Generate the combined three-model figure.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharey=True,
    )

    axes = axes.flatten()

    for ax, (model_name, composition) in zip(
        axes,
        compositions.items(),
    ):
        draw_model(ax, model_name, composition)

    # The fourth panel is not needed.
    axes[-1].axis("off")

    legend_handles = [
        Patch(
            facecolor=behavior_colors[behavior],
            label=behavior_labels[behavior],
        )
        for behavior in behaviors
        if any(
            compositions[model][behavior].sum() > 0
            for model in compositions
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(
        "Manually annotated response strategies by role and intent",
        fontsize=14,
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    combined_path = (
        output_dir / "manual_role_intent_behavior_mix.png"
    )

    fig.savefig(
        combined_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Combined figure saved to: {combined_path}")

    return compositions


generate_manual_behavior_plots(
    llama_file="llama_role_manual_validation.csv",
    gemma_file="gemma_role_manual_validation.csv",
    qwen_file="qwen_role_manual_validation.csv",
    output_dir="outputs/summary",
)