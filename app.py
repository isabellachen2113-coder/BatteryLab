import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
import re
import sys
import subprocess
from pathlib import Path

from utils.data_loader import loadMat, getBatteryCapacity, getBatteryValues

st.set_page_config(page_title="BatteryLab", layout="wide")
st.title("🔋 BatteryLab：电池健康预测仿真工坊")

# 解决中文字体与负号显示问题（macOS 常见字体）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

APP_DIR = Path(__file__).resolve().parent
CODE_REPO_DIR = APP_DIR.parent / "代码作品集" / "2.3"
DEFAULT_RUL_REPO = CODE_REPO_DIR / "RUL_prediction-main"
DEFAULT_CNN_REPO = CODE_REPO_DIR / "CNN-ASTLSTM-main"


def build_mock_rul_curve(cycles, base, decay, noise):
    # Simple synthetic curve for interactive visualization.
    trend = base * np.exp(-decay * cycles)
    jitter = np.sin(cycles / 6.0) * noise
    return np.maximum(trend + jitter, 0)


def render_astlstm_diagram():
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.set_axis_off()

    def add_box(x, y, w, h, text):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1,
            edgecolor="#4c4c4c",
            facecolor="#f2f2f2"
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    add_box(0.02, 0.25, 0.2, 0.5, "Input\n(V/I/T/C)")
    add_box(0.28, 0.25, 0.18, 0.5, "CNN\nFeature")
    add_box(0.52, 0.25, 0.18, 0.5, "ATS-LSTM\nTemporal")
    add_box(0.76, 0.25, 0.22, 0.5, "Dense\nSOH / RUL")

    ax.annotate("", xy=(0.28, 0.5), xytext=(0.22, 0.5), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.52, 0.5), xytext=(0.46, 0.5), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.76, 0.5), xytext=(0.70, 0.5), arrowprops={"arrowstyle": "->"})

    return fig


def format_py_value(value):
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def update_param_file(param_path, overrides):
    if not param_path.exists():
        return False, f"参数文件不存在: {param_path}"
    lines = param_path.read_text(encoding="utf-8").splitlines()
    found = set()
    for i, line in enumerate(lines):
        for key, value in overrides.items():
            pattern = rf"^(\s*{re.escape(key)}\s*=\s*).*$"
            match = re.match(pattern, line)
            if match:
                lines[i] = f"{match.group(1)}{format_py_value(value)}"
                found.add(key)
                break
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [key for key in overrides.keys() if key not in found]
    if missing:
        return True, f"已更新参数，但未找到: {', '.join(missing)}"
    return True, "参数已更新"


def run_external_script(script_path, work_dir, timeout_sec):
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec if timeout_sec > 0 else None
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        return exc.stdout or "", f"运行超时（{timeout_sec}s）", 124
    except Exception as exc:
        return "", f"运行失败: {exc}", 1


def find_latest_eval_dir(base_dir):
    if not base_dir.exists():
        return None
    metrics_files = sorted(
        base_dir.glob("**/eval_metrics.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not metrics_files:
        return None
    return metrics_files[0].parent


def parse_eval_metrics(metrics_path):
    metrics = {}
    if not metrics_path.exists():
        return metrics
    text = metrics_path.read_text(encoding="utf-8")
    patterns = {
        "MAE": r"Test Mean Absolute Error:\s*([0-9.]+)",
        "MSE": r"Test Mean Square Error:\s*([0-9.]+)",
        "MAPE": r"Test Mean Absolute Percentage Error:\s*([0-9.]+)",
        "RMSE": r"Test Root Mean Squared Error:\s*([0-9.]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def load_predictions(eval_dir):
    pred_path = eval_dir / "test_predict.txt"
    true_path = eval_dir / "test_true.txt"
    if not pred_path.exists() or not true_path.exists():
        return None, None
    pred = np.loadtxt(pred_path)
    true = np.loadtxt(true_path)
    return true, pred


# ---------- 侧边栏：模块选择 ----------
st.sidebar.header("🧩 模块导航")
app_mode = st.sidebar.radio("模块选择", ["数据工坊", "模型复现"], index=0)

# 定义电池MAT文件路径（请确保data文件夹内有这些文件）
mat_files = {
    "B0005": "data/B0005.mat",
    "B0006": "data/B0006.mat",
    "B0007": "data/B0007.mat",
    "B0018": "data/B0018.mat"
}


@st.cache_resource
def load_battery_data(name):
    matfile = mat_files[name]
    raw_data = loadMat(matfile)
    capacity = getBatteryCapacity(raw_data)
    charge_data = getBatteryValues(raw_data, Type="charge")
    discharge_data = getBatteryValues(raw_data, Type="discharge")
    return {
        "raw": raw_data,
        "capacity": capacity,          # [cycles, capacities]
        "charge": charge_data,
        "discharge": discharge_data
    }


if app_mode == "数据工坊":
    # 加载电池列表（只显示文件存在的电池）
    available_batteries = [name for name, path in mat_files.items() if os.path.exists(path)]
    if not available_batteries:
        st.error("未找到任何MAT文件！请将NASA数据集放在 `data/` 文件夹下。")
        st.stop()

    selected_battery = st.sidebar.selectbox(
        "1. 选择电池",
        available_batteries,
        index=0
    )

    data = load_battery_data(selected_battery)

    view_mode = st.sidebar.radio(
        "2. 选择视图",
        ["容量衰减曲线", "充电电流曲线", "放电电压曲线"]
    )

    if view_mode in ["充电电流曲线", "放电电压曲线"]:
        if view_mode == "充电电流曲线":
            max_cycles = len(data["charge"])
        else:
            max_cycles = len(data["discharge"])

        if max_cycles == 0:
            st.sidebar.warning("该电池无对应类型数据")
            selected_cycles = []
        else:
            default_cycles = list(range(min(3, max_cycles)))
            selected_cycles = st.sidebar.multiselect(
                f"选择要显示的循环序号 (0 ~ {max_cycles - 1})",
                options=list(range(max_cycles)),
                default=default_cycles
            )
    else:
        selected_cycles = []

    if view_mode == "容量衰减曲线":
        show_split = st.sidebar.checkbox("显示训练/测试划分", value=False)
        total_cycles = len(data["capacity"][0])
        if show_split:
            train_ratio = st.sidebar.slider("训练集比例 (%)", 20, 90, 70, 5)
            split_idx = int(total_cycles * train_ratio / 100)
            eol_mode = st.sidebar.radio("EOL阈值设置", ["固定值 (1.38Ah)", "动态阈值 (初始容量的80%)"])
            if eol_mode == "固定值 (1.38Ah)":
                eol_threshold = 1.38
            else:
                eol_threshold = data["capacity"][1][0] * 0.8

            eol_cycle_count = total_cycles
            for i, cap in enumerate(data["capacity"][1]):
                if cap <= eol_threshold:
                    eol_cycle_count = i + 1
                    break
            if split_idx > eol_cycle_count:
                split_idx = eol_cycle_count
                train_ratio = int(round(split_idx / total_cycles * 100)) if total_cycles else 0
                st.sidebar.warning("训练集样本数不能超过真实失效循环数，已自动调整。")
        else:
            train_ratio = None
            split_idx = None
            eol_threshold = None
    else:
        train_ratio = None
        split_idx = None
        eol_threshold = None

    st.subheader(f"📈 {selected_battery} - {view_mode}")

    fig, ax = plt.subplots(figsize=(12, 5))

    if view_mode == "容量衰减曲线":
        cycles = data["capacity"][0]
        capacities = data["capacity"][1]
        ax.plot(cycles, capacities, "b-", label="Full lifecycle", alpha=0.6, linewidth=2)
        if show_split and split_idx is not None:
            ax.plot(cycles[:split_idx], capacities[:split_idx], "g-", label=f"Train ({train_ratio}%)", linewidth=2.5)
            ax.plot(cycles[split_idx:], capacities[split_idx:], "r--", label=f"Test ({100 - train_ratio}%)", linewidth=2)
            ax.axhline(y=eol_threshold, color="purple", linestyle=":", linewidth=2, label=f"EOL threshold = {eol_threshold:.2f}Ah")
            ax.axvline(x=split_idx, color="gray", linestyle="--", alpha=0.7)
            ax.text(split_idx + 2, ax.get_ylim()[1] * 0.9, f"Split\n{split_idx} cycles", fontsize=9)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity (Ah)")
        ax.set_title(f"{selected_battery} Capacity degradation")

    elif view_mode == "充电电流曲线":
        if not selected_cycles:
            ax.text(0.5, 0.5, "请从左侧选择要显示的循环", ha="center", va="center", transform=ax.transAxes)
        else:
            color_list = ["b", "g", "r", "c", "m", "y"]
            for i, cycle_idx in enumerate(selected_cycles):
                if cycle_idx < len(data["charge"]):
                    cycle_data = data["charge"][cycle_idx]
                    ax.plot(cycle_data["Time"], cycle_data["Current_measured"],
                            color=color_list[i % len(color_list)],
                            label=f"Cycle {cycle_idx}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Current (A)")
            ax.set_title(f"{selected_battery} Charge current (multiple cycles)")

    elif view_mode == "放电电压曲线":
        if not selected_cycles:
            ax.text(0.5, 0.5, "请从左侧选择要显示的循环", ha="center", va="center", transform=ax.transAxes)
        else:
            color_list = ["b", "g", "r", "c", "m", "y"]
            for i, cycle_idx in enumerate(selected_cycles):
                if cycle_idx < len(data["discharge"]):
                    cycle_data = data["discharge"][cycle_idx]
                    ax.plot(cycle_data["Time"], cycle_data["Voltage_measured"],
                            color=color_list[i % len(color_list)],
                            label=f"Cycle {cycle_idx}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"{selected_battery} Discharge voltage (multiple cycles)")

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    if view_mode == "容量衰减曲线":
        if show_split and split_idx is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总循环数", len(data["capacity"][0]))
            with col2:
                st.metric("训练集样本", split_idx)
            with col3:
                st.metric("测试集样本", len(data["capacity"][0]) - split_idx)
            with col4:
                st.metric("初始容量", f"{data['capacity'][1][0]:.2f} Ah")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总循环数", len(data["capacity"][0]))
            with col2:
                st.metric("初始容量", f"{data['capacity'][1][0]:.2f} Ah")
    else:
        total_cycles = len(data["charge"]) if view_mode == "充电电流曲线" else len(data["discharge"])
        st.info(f"该电池共有 {total_cycles} 个{'充电' if view_mode == '充电电流曲线' else '放电'}循环数据")

    st.caption("数据来源：NASA PCoE 公开数据集。")

elif app_mode == "模型复现":
    st.subheader("🧪 模型复现：CNN-ASTLSTM / RUL_prediction")
    st.info("仅展示预训练结果与参数模拟，不需要学生配置环境。")

    rul_repo_dir = DEFAULT_RUL_REPO
    cnn_repo_dir = DEFAULT_CNN_REPO

    col_a, col_b = st.columns(2)
    with col_a:
        st.link_button("RUL_prediction (GitHub)", "https://github.com/huzaifi18/RUL_prediction")
    with col_b:
        st.link_button("CNN-ASTLSTM (GitHub)", "https://github.com/Lipenghua-CQ/CNN-ASTLSTM")

    model_family = st.sidebar.selectbox(
        "1. 选择模型来源",
        ["RUL_prediction", "CNN-ASTLSTM"],
        index=0
    )

    run_tab, preview_tab = st.tabs(["结果可视化", "交互式可视化"])

    with run_tab:
        if model_family == "RUL_prediction":
            st.markdown("**RUL_prediction 复现（4个模型）**")
            st.markdown("""
**学习框架（模型原理）**
- 任务：利用充电/放电曲线中的 V/I/T/C 特征预测容量衰减与 RUL。
- SC/MC：SC=单通道（例如仅 V 或 V+C），MC=多通道（V/I/T/C 组合）。
- LSTM：建模时间序列依赖；CNN：提取局部形状特征；CNN+LSTM：先抽特征再建模序列。
- 核函数：本系列模型为深度神经网络，不使用核函数（Kernel-based 方法）。
""")

            framework_img = rul_repo_dir / "framework.png"
            if framework_img.exists():
                st.image(str(framework_img), caption="RUL_prediction 模型结构示意图", use_column_width=True)
            else:
                st.info("未找到结构示意图（framework.png），可在仓库根目录添加。")

            rul_models = {
                "SC-LSTM": {"script": "SC-LSTM.py", "param": "param_VC_C.py"},
                "MC-LSTM": {"script": "MC-LSTM.py", "param": "param_VITC_C.py"},
                "SC-CNN+LSTM": {"script": "SC-CNN+LSTM.py", "param": "param_V_CNN_C_LSTM.py"},
                "MC-SCNN+LSTM": {"script": "MC-SCNN+LSTM.py", "param": "param_separated.py"}
            }
            model_name = st.sidebar.selectbox("2. 选择具体模型", list(rul_models.keys()))

            rul_desc = {
                "SC-LSTM": "单通道 LSTM，使用单一特征序列进行容量预测。",
                "MC-LSTM": "多通道 LSTM，融合 V/I/T/C 等多特征。",
                "SC-CNN+LSTM": "单通道 CNN + LSTM，CNN 提取局部模式，LSTM 建模时序。",
                "MC-SCNN+LSTM": "多通道分支 CNN + LSTM，多特征分别卷积后融合。"
            }
            st.info(rul_desc.get(model_name, ""))

            eval_dir = find_latest_eval_dir(rul_repo_dir / "saved")
            if eval_dir:
                metrics = parse_eval_metrics(eval_dir / "eval_metrics.txt")
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics.get('MAE', 0.0):.4f}")
                    with col2:
                        st.metric("MSE", f"{metrics.get('MSE', 0.0):.4f}")
                    with col3:
                        st.metric("MAPE", f"{metrics.get('MAPE', 0.0):.4f}")
                    with col4:
                        st.metric("RMSE", f"{metrics.get('RMSE', 0.0):.4f}")

                true_vals, pred_vals = load_predictions(eval_dir)
                if true_vals is not None and pred_vals is not None:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(true_vals, label="True", color="#2ca02c")
                    ax.plot(pred_vals, label="Pred", color="#1f77b4")
                    ax.set_title("Test Prediction vs True")
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Capacity")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            else:
                st.info("未检测到预训练结果，请先在本地运行模型生成结果文件。")

        else:
            st.markdown("**CNN-ASTLSTM 复现**")
            st.markdown("""
**学习框架（模型原理）**
- 任务：SOH 估计与 RUL 预测（容量曲线建模）。
- CNN：提取局部模式；ATS-LSTM：注意力时序单元，强化关键时间片的贡献。
- 核函数：该模型为深度神经网络结构，不涉及显式核函数。
""")
            st.pyplot(render_astlstm_diagram())
            st.warning("该仓库依赖 TensorFlow 1.9 / Keras 2.1.5，建议在独立环境中运行。")

            st.info("此模块用于学习结构与原理，训练请在独立环境完成。")

    with preview_tab:
        st.markdown("**参数变动可视化（模拟曲线）**")
        st.sidebar.markdown("---")
        st.sidebar.subheader("可视化参数")
        preview_epochs = st.sidebar.slider("训练轮数（预览）", 1, 200, 30, 1)
        preview_lr = st.sidebar.number_input("学习率（预览）", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f")
        preview_batch = st.sidebar.select_slider("Batch size（预览）", options=[8, 16, 32, 64, 128], value=32)
        preview_layers = st.sidebar.slider("层数（预览）", 1, 6, 3, 1)
        preview_hidden = st.sidebar.slider("隐藏维度（预览）", 16, 256, 64, 8)
        preview_eol = st.sidebar.slider("EOL 阈值 (Ah)", 0.6, 1.6, 1.0, 0.05)

        fig, ax = plt.subplots(figsize=(12, 5))
        cycle_idx = np.arange(1, 151)
        base = 1.55 - (preview_layers * 0.03)
        decay = 0.006 + (preview_lr * 5)
        noise = 0.02 + (preview_hidden / 1024)
        mock_curve = build_mock_rul_curve(cycle_idx, base=base, decay=decay, noise=noise)

        ax.plot(cycle_idx, mock_curve, color="#1f77b4", linewidth=2, label="Preview (mock)")
        ax.axhline(y=preview_eol, color="gray", linestyle="--", linewidth=1.5, label="EOL threshold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity (Ah)")
        ax.set_title("Mock degradation curve (interactive)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Epochs", preview_epochs)
        with col2:
            st.metric("LR", f"{preview_lr:.5f}")
        with col3:
            st.metric("Batch", preview_batch)
        with col4:
            st.metric("Layers", preview_layers)


    st.markdown("**课堂最简流程（建议）**")
    st.markdown("""
1. 你在本地或 Colab 运行模型，生成 `test_predict.txt`/`test_true.txt` 和 `eval_metrics.txt`。
2. 将结果文件提交到仓库（或上传到指定目录）。
3. 学生在云端页面只做参数模拟与结果对比，不需要配置环境。
""")
