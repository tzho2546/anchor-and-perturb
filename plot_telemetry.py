import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 实验分组配置
experiment_groups = {
    "Baseline (Monotonic QMIX)": {
        "files": ["qmix_vanilla_seed34_2s3z-20260922-142443.json"],
        "color": "#e74c3c"
    },
    "Anchor-and-Perturb (AnP)": {
        "files": ["qmix_anp_seed34_2s3z-20260922-162010.json"],
        "color": "#27ae60"
    }
}


# 2. Tensorboard 风格的指数移动平均 (EMA) 平滑函数
def smooth_tensorboard(scalars, weight=0.85):
    """
    weight 介于 0 和 1 之间。值越大曲线越平滑 (推荐 0.8 - 0.95)。
    """
    if len(scalars) == 0:
        return np.array([])
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


# 3. 读取、平滑并对齐数据
def load_group_data(file_list, metric, smooth_weight=0.85):
    all_runs = []
    episodes = None
    for f in file_list:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 提取指定的 metric
                values = [item[metric] for item in data]
                # 在塞入数组前，对这条单 run 曲线做平滑处理
                smoothed_values = smooth_tensorboard(values, weight=smooth_weight)
                all_runs.append(smoothed_values)

                if episodes is None:
                    episodes = [item["episode"] for item in data]
        else:
            print(f"Warning: {f} not found, skipping...")

    if not all_runs:
        return None, None, None

    all_runs = np.array(all_runs)
    mean_vals = np.mean(all_runs, axis=0)
    std_vals = np.std(all_runs, axis=0)

    return episodes, mean_vals, std_vals


# 4. 创建双子图布局
fig, (ax_win, ax_rew) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, dpi=300)

for group_name, config in experiment_groups.items():
    files = config["files"]
    color = config["color"]

    # 绘制胜率及其平滑后的标准差
    eps, win_mean, win_std = load_group_data(files, "eval_win_rate", smooth_weight=0.85)
    if eps is not None:
        ax_win.plot(eps, win_mean, label=group_name, color=color, linewidth=2)
        ax_win.fill_between(eps,
                            np.clip(win_mean - win_std, 0, 1),
                            np.clip(win_mean + win_std, 0, 1),
                            color=color, alpha=0.2)

    # 绘制回报及其平滑后的标准差
    _, rew_mean, rew_std = load_group_data(files, "eval_reward", smooth_weight=0.85)
    if rew_mean is not None:
        ax_rew.plot(eps, rew_mean, label=group_name, color=color, linewidth=2)
        ax_rew.fill_between(eps, rew_mean - rew_std, rew_mean + rew_std, color=color, alpha=0.2)

# 5. 图表轴与排版美化
ax_win.set_ylabel("Evaluation Win Rate", fontsize=11, fontweight="bold")
ax_win.set_ylim(-0.05, 1.05)
ax_win.grid(True, linestyle="--", alpha=0.45)
ax_win.legend(loc="upper left", frameon=True, fontsize=9, framealpha=0.9)

ax_rew.set_ylabel("Evaluation Team Reward", fontsize=11, fontweight="bold")
ax_rew.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
ax_rew.set_xlim(0, max(eps) + 50 if eps is not None else 3050)
ax_rew.grid(True, linestyle="--", alpha=0.45)

plt.suptitle("MARL Performance: AnP vs. Baseline (Smoothed Mean ± Std)", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()

# 6. 导出
output_pdf = "fig4_telemetry_smoothed.pdf"
output_png = "fig4_telemetry_smoothed.png"
plt.savefig(output_pdf, bbox_inches="tight")
plt.savefig(output_png, bbox_inches="tight")
print(f"Smoothed plotting complete! Saved to {output_pdf} and {output_png}")