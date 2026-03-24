# AutoFigure-Edit 核心工作流 (Workflow) 梳理

## 1. 核心流程概述
AutoFigure-Edit 的核心目标是将一段纯文本（Paper Method）或者一张不可编辑的像素图，转换为**结构化、可深度编辑的纯矢量 SVG 格式**。
整个流水线分为以下五个关键步骤：

1. **Step 1: 图像生成 (Text to Image)**
   - **输入**：用户提供的描述文本（Method Text）。
   - **输出**：一张高质量但不可编辑的像素底图 `figure.png`。
   - *(注：如果直接上传了参考图，可跳过此步直接使用参考图作为 `figure.png`)*。

2. **Step 2: 目标检测与标注 (SAM3 + Layout Annotation)**
   - **动作**：使用 SAM3 模型识别 `figure.png` 中的核心实体（如 icon, robot, person）。
   - **输出**：
     - `boxlib.json`：记录所有被识别出的实体的精确像素坐标。
     - `samed.png`：在原图上，用带有编号（如 `<AF>01`）的灰色实心矩形块遮盖住这些实体。

3. **Step 3: 实体提取去背 (Crop & Remove Background)**
   - **动作**：根据 `boxlib.json` 的坐标，把 `figure.png` 中的各个实体裁剪下来，并使用 RMBG 算法去掉背景，变成透明的独立图标。
   - **输出**：一系列透明的 `icon_AF01_nobg.png` 等文件。

4. **Step 4: 多模态重绘 SVG 模板 (Multimodal SVG Reconstruction)**
   - **动作**：将 `figure.png` 和 `samed.png` 喂给多模态大模型。
   - **指令**：“请你用纯 SVG 代码（`<rect>`, `<path>`, `<text>`），完美重绘这张图里的线条、框线和文字。至于图标部分不要画，直接在对应位置画上和 `samed.png` 一模一样的带有 `<AF>01` 标签的灰色占位符。”
   - **输出**：一个纯代码的 `template.svg`。

5. **Step 5: 图标注入与最终合成 (Icon Injection)**
   - **动作**：Python 后端解析 `template.svg`。找到所有带有 `<AF>01` 标签的灰色占位符 `<g>` 标签。
   - **替换**：将这些占位符删除，并在该坐标位置插入 `<image>` 标签，把 Step 3 抠出来的无背景透明图标 `icon_AF01_nobg.png` 嵌进去。
   - **输出**：最终在前端展示的可编辑的 `final.svg`。