# 验证记录

验证日期：2026-10-31。

环境：Windows，Python 3.12.2，NumPy 2.5.4，Pillow 13.0.0，Streamlit 1.66.1，pandas 3.1.0。

- 5 项 unittest 测试全部通过：满秩重建、理论误差一致性、存储公式、输入处理、因子导出还原。
- Streamlit AppTest 通过：初始页面、k=0、k=256 满秩重建、灰度模式、最长边切换和 float64 存储精度切换，无应用异常。
- Python 源码语法检查通过。

界面验证使用 Streamlit 自动化组件测试，未进行浏览器截图级视觉验收。其他操作系统和 Python 版本未在本次环境实测。
