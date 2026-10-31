import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from core import (prepare_image, demo_image, decompose, reconstruct, display_pixels,
                  metrics, storage, theoretical_curve, png_bytes, factor_bytes)

st.set_page_config(page_title="图像压缩实验室", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background: #f5f7fa;}
h1 {letter-spacing: -1.5px;}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:14px;border:1px solid #e3e8ef;}
</style>""", unsafe_allow_html=True)
st.caption("LINEAR ALGEBRA / 交互实验 01")
st.title("图像压缩实验室")
st.write("一张图片，多少个奇异值才够？拖动滑块，观察细节、误差与存储成本的变化。")

@st.cache_data(max_entries=4, show_spinner=False)
def analyze(data, limit, grayscale):
    pixels, original_size = prepare_image(data, limit, grayscale)
    return pixels, original_size, decompose(pixels)

with st.sidebar:
    st.header("实验设置")
    upload = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp", "bmp"])
    st.caption("未上传时使用内置示例 · 最大 20 MB / 2500 万像素")
    mode = st.radio("色彩模式", ["RGB 彩色", "灰度"])
    limit = st.select_slider("处理图片的最长边", options=[128, 256, 384, 512, 768], value=384)
    precision = st.selectbox("因子存储精度", ["float32 · 4 字节", "float64 · 8 字节"])
    st.caption("首次计算需要进行 SVD；同一图片拖动滑块时复用分解结果。")

data = upload.getvalue() if upload else demo_image()
gray = mode == "灰度"
try:
    with st.spinner("正在分解图片矩阵…"):
        pixels, original_size, factors = analyze(data, limit, gray)
except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
    st.error(f"无法读取图片：{exc}")
    st.stop()
except np.linalg.LinAlgError:
    st.error("SVD 未收敛，请尝试其他图片或更小的处理尺寸。")
    st.stop()

h, w = pixels.shape[:2]
rank_max = min(h, w)
st.caption(f"上传尺寸 {original_size[0]} × {original_size[1]} → 实验尺寸 {w} × {h} · "
           "透明背景合成为白色；左右图与误差均基于同一实验尺寸。")
key = hashlib.sha256(data + f"{limit}{gray}".encode()).hexdigest()[:16]
k = st.slider("保留的奇异值数量 k（每个通道）", 0, rank_max,
              min(30, rank_max), key=key)
reconstruction = reconstruct(factors, k, gray)
shown = display_pixels(reconstruction)
mse, psnr, mae = metrics(pixels, shown)
byte_width = 4 if precision.startswith("float32") else 8
raw, values, compressed = storage(pixels.shape, k, byte_width)
curve = pd.DataFrame(theoretical_curve(factors, pixels.shape, byte_width))
energy = curve.loc[k, "保留能量(%)"]

left, right = st.columns(2)
with left:
    st.subheader("原图 · 实验基准")
    st.image(pixels, width="stretch", clamp=True)
with right:
    st.subheader(f"SVD 重建 · k = {k}")
    st.image(shown, width="stretch", clamp=True)

a, b, c, d = st.columns(4)
a.metric("重建误差 MSE ↓", f"{mse:.3f}")
b.metric("峰值信噪比 PSNR ↑", "∞ dB" if np.isinf(psnr) else f"{psnr:.2f} dB")
c.metric("保留奇异值能量", f"{energy:.2f}%")
d.metric("理论存储 / 原始像素", f"{100*compressed/raw:.1f}%")
st.caption(f"平均绝对误差 MAE：{mae:.3f}。MSE、MAE 和 PSNR 比较原始 uint8 像素与裁剪、四舍五入后的显示图；能量不是视觉质量评分。")

tab1, tab2, tab3 = st.tabs(["质量与存储", "误差与奇异值", "原理与导出"])
with tab1:
    st.subheader("多保留一个奇异值，会带来什么？")
    p, q = st.columns(2)
    with p:
        st.write("理论误差随 k 的变化")
        st.line_chart(curve.set_index("k")[["理论MSE"]])
    with q:
        st.write("存储成本随 k 的变化")
        st.line_chart(curve.set_index("k")[["占原始像素(%)"]])
    st.caption("曲线误差来自舍弃奇异值的平方和，针对未裁剪、未量化的浮点重建；与上面的显示图误差略有差别。")
    st.table(pd.DataFrame({"项目": ["原始像素（uint8）", "SVD 因子（理论）", "上传文件 / 示例 PNG"],
                           "字节数": [raw, compressed, len(data)]}))
    st.info(f"当前存储 {values:,} 个因子数值，每个 {byte_width} 字节。"
            "理论存储不包含文件头和元数据；上传文件可能已有 JPEG/PNG 编码，不能与像素存储混为一谈。")
    if compressed >= raw:
        st.warning("当前 k 和存储精度下，SVD 因子不比原始像素省空间。可减少 k，观察质量变化。")
with tab2:
    p, q = st.columns(2)
    with p:
        st.write("绝对误差图 · 固定放大 4 倍")
        error = np.abs(pixels.astype(float) - shown.astype(float))
        if error.ndim == 3:
            error = error.mean(axis=2)
        st.image(np.uint8(np.clip(error*4, 0, 255)), width="stretch")
        st.caption("越亮表示误差越大；彩色图按三个通道取平均。")
    with q:
        st.write("奇异值谱")
        names = ["灰度"] if gray else ["R", "G", "B"]
        spectrum = pd.DataFrame({name: f[1] for name, f in zip(names, factors)},
                                index=np.arange(1, rank_max+1))
        st.line_chart(spectrum)
with tab3:
    st.latex(r"A=U\Sigma V^T,\qquad A_k=U_{[:,1:k]}\Sigma_{1:k}V^T_{[1:k,:]}")
    st.write("每个通道是一张 m × n 的矩阵。保留最大的 k 个奇异值，就得到秩至多为 k 的近似。"
             "RGB 三通道分别分解，使用相同的 k。k=0 得到全黑图；k=min(m,n) 可近乎精确重建。")
    st.latex(r"N_{\mathrm{original}}=cmn,\quad N_{\mathrm{SVD}}=ck(m+n+1)")
    st.write("原始 uint8 像素每项 1 字节；浮点因子每项 4 或 8 字节。"
             "计算使用 float64；存储精度只影响理论字节数和导出的因子，预览不模拟 float32 量化。")
    st.write("建议实验：分别选风景、文字和纹理图，在 k=5、10、20、50 记录误差和存储比例，"
             "比较哪些图像更容易用低秩近似表示。")
    st.download_button("下载重建图 PNG", png_bytes(shown), f"reconstructed_k{k}.png", "image/png")
    st.download_button("下载全量实验数据 CSV", curve.to_csv(index=False).encode("utf-8-sig"),
                       "svd_experiment.csv", "text/csv")
    st.download_button("下载截断 SVD 因子 NPZ", factor_bytes(factors, k, pixels.shape,
                       "float32" if byte_width == 4 else "float64"), f"svd_k{k}.npz", "application/octet-stream")
    st.caption("PNG 是预览图编码，不是 SVD 因子文件。NPZ 包含元数据和容器开销，实际大小会大于理论因子字节数。")
