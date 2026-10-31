# 图像压缩实验室
使用 NumPy 实现逐通道 SVD，通过 Streamlit 本地网页上传图片、调节奇异值数量、比较质量与存储成本。自带生成式几何风景示例，无需额外下载素材。

## 快速启动

需要 Python 3.10 或更高版本，建议 Python 3.12。首次安装依赖需要网络。

**Windows：** 解压 ZIP 后进入项目文件夹，双击 `start.bat`。需事先安装 Python 并勾选“Add Python to PATH”。启动脚本会创建独立环境、安装依赖并打开浏览器；保持命令窗口开启，按 Ctrl+C 停止。若浏览器未自动打开，访问 http://localhost:8501 。

**手动启动（Windows PowerShell）：**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py --server.address localhost
```

**macOS / Linux：**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address localhost
```

不要直接用 `python app.py` 启动界面。端口冲突时，在启动命令末尾添加 `--server.port 8502`。安装失败请确认网络和 Python 版本；项目不包含 Python 解释器或第三方依赖。

## 如何使用

1. 打开应用，先用内置示例体验，或上传 PNG / JPEG / WebP / BMP。
2. 选择 RGB 或灰度及处理最长边。默认最长边 384，最大 768；等比缩小，不放大小图片。
3. 拖动 k 滑块，比较左右图和 MSE、PSNR、保留能量与存储百分比。
4. 查看“质量与存储”曲线、“误差与奇异值”及数学原理。
5. 导出重建 PNG、所有 k 的理论数据 CSV 或当前截断因子 NPZ。

上传图像仅在本地 Python 进程中处理。最多 20 MB、2500 万像素；透明图像合成为白色，EXIF 方向自动校正，多帧文件只取第一帧。不支持保留透明度或色彩配置文件的专业色彩管理。

所有指标都基于预处理后的实验图像，不衡量缩放或灰度转换带来的损失。同一图像和设置下缓存 SVD，拖动 k 不会重新分解；更换图片、模式或尺寸会重新计算。大尺寸首次分解可能较慢，可降到 128 / 256。

## 数学与存储口径

对每个 m×n 通道矩阵 A，调用 `np.linalg.svd(A, full_matrices=False)`：

```python
U, s, Vh = np.linalg.svd(A, full_matrices=False)
A_k = (U[:, :k] * s[:k]) @ Vh[:k, :]
```

奇异值按降序排列。RGB 对三个通道分别分解并保留同一个 k；灰度只有一个通道。k=0 是黑图；k=min(m,n) 在浮点误差范围内恢复实验图像。

- 原始 uint8 像素：`c*m*n` 字节。
- 因子数值数：`c*k*(m+n+1)`，包含 U、奇异值向量和 Vh，不存储完整对角矩阵。
- 理论因子字节数：`c*k*(m+n+1)*b`，b 为 4（float32）或 8（float64）。
- 只有 `k < m*n/[b*(m+n+1)]` 时，理论因子存储才小于原始 uint8 像素。
- “理论存储 / 原始像素”越小越省空间，超过 100% 表示没有节省。
- JPEG/PNG 文件可能已被编码压缩，不是原始像素存储。导出的 PNG 也不等于 SVD 压缩数据。
- NPZ 保存实际因子、shape 和 k，存在数组头、容器和元数据开销，因此比纯因子理论大小大。

预览计算统一使用 float64；精度选项只控制理论存储和因子导出，float32 导出还原后可能出现微小量化差异。

**显示图指标：** 先将浮点重建裁剪到 [0,255] 并四舍五入为 uint8，再计算 MSE、MAE 和 PSNR。MSE 为像素差平方的平均值，MAE 为绝对差均值；`PSNR = 10*log10(255²/MSE)`，MSE=0 时为无穷大。

**理论曲线：** 未裁剪浮点近似的 MSE 为所有通道被舍弃奇异值平方之和除以 `c*m*n`。与实际显示图误差口径不同。保留能量等于保留奇异值平方和占总平方和的比例；全黑图总能量为零，约定显示 100%。能量高并不保证文字细节或主观视觉质量好。

## 恢复因子文件

使用同一 Python 环境：

```powershell
.venv\Scripts\python restore.py svd_k30.npz restored.png
```

macOS / Linux 将解释器路径换成 `.venv/bin/python`。

## 实验建议与 1–2 天安排

第一天：理解矩阵表示、实现 SVD 与截断重建、验证满秩恢复、搭建上传和滑块界面。

第二天：比较风景、文字、随机纹理图，导出数据，整理结论。对每张图固定处理尺寸，记录 k=5、10、20、50 等数值下的 MSE、PSNR 和存储成本；小图的 k 不超过 min(m,n)。

可探究：同样的存储预算下，哪类图像更清晰？保留 95% 能量是否足够？float32 相较 float64 如何改变存储盈亏点？为什么 SVD 对噪声纹理效果较差？报告中注明这是低秩近似教学实验，未实现量化编码、熵编码或 JPEG 级压缩器。

## 文件与测试

- `app.py`：交互界面与结果展示。
- `core.py`：图像预处理、NumPy SVD、误差、存储和导出。
- `restore.py`：将 NPZ 因子重建为 PNG。
- `test_core.py`：数值正确性及输入处理测试。
- `requirements.txt`：依赖范围。
- `start.bat`：Windows 启动脚本。

```powershell
.venv\Scripts\python -m unittest -v test_core.py
```

测试覆盖 RGB / 灰度 / 单像素边长的满秩恢复、奇异值尾部能量与实际误差一致性、误差单调性、理论因子字节数、零矩阵、透明背景和无效图片。

## 参考文档

- NumPy SVD：https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- Streamlit 上传：https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit 缓存：https://docs.streamlit.io/develop/concepts/architecture/caching
