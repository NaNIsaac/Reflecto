import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


def plot_profile(df, png_path, sample_label, styles):
    """Create the electron density profile graph with the given style parameters."""

    fig, ax = plt.subplots(figsize=(8, 6))

    y_values = df["Electron Density(e/cm³)"] / 1e23
    ax.plot(df["z(Å)"], y_values, color="black", lw=1.5, label=sample_label)

    ax.set_title(
        "Electron Density Profile",
        fontsize=styles["title_fontsize"],
        weight=styles["title_fontweight"],
        pad=15,
    )

    ax.set_xlabel(
        r"Distance from Substrate (Å)",
        fontsize=styles["xlabel_fontsize"],
        weight=styles["xlabel_fontweight"],
    )
    ax.set_ylabel(
        r"Electron Density ($e/\mathrm{cm}^3$)",
        fontsize=styles["ylabel_fontsize"],
        weight=styles["ylabel_fontweight"],
    )

    ax.annotate(
        r"$\times10^{23}$",
        xy=(0, 1),
        xytext=(5, 10),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=15,
        ha="right",
        va="center",
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=styles["tick_labelsize"],
        width=styles["tick_linewidth"],
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=styles["minor_tick_linewidth"],
        direction="in",
        top=True,
        right=True,
    )

    if sample_label:
        legend = ax.legend(loc="upper right", fontsize=styles["legend_fontsize"], frameon=True)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(styles["legend_border_linewidth"])

    for spine in ax.spines.values():
        spine.set_linewidth(styles["spine_linewidth"])
        spine.set_color("black")

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=styles["grid_linewidth"],
        alpha=styles["grid_alpha"],
    )

    ax.set_xlim(df["z(Å)"].min(), df["z(Å)"].max())

    if styles["y_axis_limits"]:
        ax.set_ylim(styles["y_axis_limits"][0], styles["y_axis_limits"][1])
    else:
        ax.set_ylim(bottom=0)

    if styles["y_tick_increment"]:
        ax.yaxis.set_major_locator(MultipleLocator(styles["y_tick_increment"]))

    plt.tight_layout()

    messagebox.showinfo(
        "그래프 편집 안내",
        "그래프 창이 나타납니다.\n\n"
        "창 하단의 도구 모음을 사용하여 확대/축소, 이동, 저장이 가능합니다.",
    )
    plt.show()

    try:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"그래프 자동 저장 완료: {png_path}")
    except Exception as e:
        print(f"그래프 파일 저장 중 오류 발생: {e}")


def convert_sld(file_path):
    base_directory = os.path.dirname(file_path)
    output_folder = os.path.join(base_directory, "electron_density_results")
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(file_path)
    stem = os.path.splitext(filename)[0]

    safe_tag = (
        stem.replace("#", "").split("sld000")[0].strip("_ ")
        if "sld000" in stem
        else stem.replace("#", "").strip()
    )
    if not safe_tag:
        safe_tag = "data"

    xlsx_path = os.path.join(output_folder, f"{safe_tag}_electron_density.xlsx")
    png_path = os.path.join(output_folder, f"{safe_tag}_electron_density.png")

    z_vals, ed_a3, ed_cm3 = [], [], []
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    z_vals.append(float(parts[0]))
                    sld = float(parts[1])
                    ed_a3.append(sld)
                    ed_cm3.append(sld * 1e24)

        if not z_vals:
            messagebox.showwarning("오류", "파일에 유효한 데이터가 없습니다.")
            return None, None, None

        df = pd.DataFrame(
            {"z(Å)": z_vals, "Electron Density(e/Å³)": ed_a3, "Electron Density(e/cm³)": ed_cm3}
        )
        df.to_excel(xlsx_path, index=False)
        print(f"엑셀 파일 저장 완료: {xlsx_path}")
        return df, png_path, safe_tag

    except Exception as e:
        messagebox.showerror("파일 처리 오류", str(e))
        return None, None, None


def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="변환할 SLD 데이터 파일 선택",
        filetypes=[("SLD dat", "*sld000.dat"), ("dat", "*.dat"), ("모든 파일", "*")],
    )
    if not file_path:
        print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    df, png_path, safe_tag = convert_sld(file_path)
    if df is None:
        return

    if messagebox.askyesno(
        "그래프 생성",
        "엑셀 변환이 완료되었습니다.\n결과는 'electron_density_results' 폴더에 저장됩니다.\n\n그래프를 생성하시겠습니까?",
    ):
        default_name = f"#{safe_tag}"
        sample_label = simpledialog.askstring(
            "Sample Name", "그래프 범례에 표시될 샘플 이름을 입력하세요:", initialvalue=default_name
        )
        if sample_label is None:
            sample_label = default_name

        # Upper limit of the y axis; change this value to rescale
        Y_AXIS_MAX = 27

        styles = {
            "title_fontsize": 16,
            "title_fontweight": "bold",
            "xlabel_fontsize": 14,
            "xlabel_fontweight": "bold",
            "ylabel_fontsize": 14,
            "ylabel_fontweight": "bold",
            "tick_labelsize": 12,
            "tick_linewidth": 1.5,
            "minor_tick_linewidth": 1.0,
            "legend_fontsize": 14,
            "legend_border_linewidth": 1.5,
            "spine_linewidth": 2.0,
            "grid_linewidth": 1.0,
            "grid_alpha": 0.8,
            "y_axis_limits": [0, Y_AXIS_MAX],
            "y_tick_increment": 2.5,
        }

        plot_profile(df, png_path, sample_label, styles)
        messagebox.showinfo("완료", "작업이 모두 완료되었습니다.")
    else:
        messagebox.showinfo("완료", "엑셀 파일 저장이 완료되었습니다.")


if __name__ == "__main__":
    main()
