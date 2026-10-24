import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_xrr_data(input_file_path, output_image_file, sample_label):
    """Read an XRR .dat file and plot the data with its fit in the requested style."""
    try:
        df = pd.read_csv(
            input_file_path,
            comment="#",
            delim_whitespace=True,
            header=None,
            usecols=[0, 1, 2],
            names=["x", "fit", "data"],
        )
        if df.empty:
            print(f"파일에 유효한 데이터가 없습니다: {input_file_path}")
            return
        df.dropna(subset=["x", "data"], inplace=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            df["x"],
            df["data"],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=6,
            linestyle="None",
            label=sample_label,
        )
        ax.plot(df["x"], df["fit"], color="red", linewidth=2.5, label=f"{sample_label} FIT")

        ax.set_yscale("log")

        ax.set_xlabel("2theta (°)", fontsize=16, weight="bold")
        ax.set_ylabel("Intensity (Arb.units)", fontsize=16, weight="bold")

        ax.tick_params(
            axis="both", which="major", labelsize=16, direction="in", top=True, right=True, width=2
        )
        ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, width=1.5)

        ax.set_yticklabels([])

        legend = ax.legend(loc="upper right", fontsize=14, frameon=True)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(2)

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color("black")

        # Optional tick control; uncomment one:
        # ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # cap the number of ticks
        # ax.set_xticks([0, 2, 4, 6, 8, 10])  # set tick positions explicitly

        plt.tight_layout()

        safe_label = sample_label.replace("#", "")
        output_image_file = os.path.join(
            os.path.dirname(input_file_path), f"{safe_label}_xrr_fit_plot.png"
        )

        plt.show()

        fig.savefig(output_image_file, dpi=300)
        print(f"그래프 저장 완료: {output_image_file}")

    except Exception as e:
        print(f"그래프 생성 중 오류 발생 ({os.path.basename(output_image_file)}): {str(e)}")


def get_sample_label(file_path, default_label):
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askstring(
        title=f"샘플 이름 입력 - {os.path.basename(file_path)}",
        prompt=f"파일: {os.path.basename(file_path)}\n\n그래프에 표시할 샘플 이름을 입력하세요:",
        initialvalue=default_label,
    )
    root.destroy()
    return user_input.strip() if user_input and user_input.strip() else default_label


def main():
    root = tk.Tk()
    root.withdraw()

    input_files = filedialog.askopenfilenames(
        title="플롯할 XRR 데이터 파일들 선택 (여러 개 선택 가능)",
        filetypes=[("Data files", "*.dat"), ("Text files", "*.txt"), ("All files", "*")],
    )

    if not input_files:
        print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        match_digit = re.search(r"(\d+)", base_name)
        default_label = match_digit.group(1) if match_digit else "data"

        sample_label = get_sample_label(input_file, default_label)

        output_image_file = os.path.join(
            os.path.dirname(input_file), f"{sample_label.replace('#', '')}_xrr_fit_plot.png"
        )
        plot_xrr_data(input_file, output_image_file, sample_label)
        print("-" * 30)


if __name__ == "__main__":
    main()
