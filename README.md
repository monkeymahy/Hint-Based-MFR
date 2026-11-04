# Hint-Based MFR

基于 PythonOCC 的 STEP 制造特征识别程序，参考论文 Hint-based generic shape feature recognition from three-dimensional B-rep models。本文以当前 `mfr_recognizer/` 实现为准。

## 环境配置

使用 Miniforge 或 Conda，在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda run -n mfr python -c "from OCC.Core.BRep import BRep_Tool; print('OCC ok')"
```

环境使用 Python 3.12、pythonocc-core 7.9 系列，OCCT 等动态库由 conda-forge 配套安装；其他运行依赖为标准库。配置文件约束主要版本，不是跨平台精确锁定文件。

更新环境：`conda env update -n mfr -f environment.yml`。

Windows 如遇 `DLL load failed`，使用 `conda run -n mfr` 或先 `conda activate mfr`。如找不到 `OCC`，检查是否使用该环境的 Python。

## 运行

当前脚本采用扁平导入，以下命令均先进入 `mfr_recognizer/`；从仓库根目录使用 `python -m mfr_recognizer.cli` 不适用于当前导入结构。

```powershell
cd mfr_recognizer
conda run -n mfr python -m cli --help
conda run -n mfr python -m cli "../data/step/01010028.step" --verbose
conda run -n mfr python -m cli "../data/step/01010028.step" --mode full --json "result.json"
conda run -n mfr python evaluate.py --dataset "../data" --details
conda run -n mfr python batch_predict.py --step-dir "../data/step" --out-dir "../data/pred_label" --mode full --overwrite
```

数据集目录应包含 `step/` 和 `label/`，面级真值文件为与 STEP 同名的 JSON 标签列表。评估输出面准确率和混淆矩阵，不评估实例匹配准确率。

CLI 支持 `--json`、`--verbose`、`--mode labels|full`、`--face-index-base 0|1`。批量预测还支持 `--threads`（别名 `--workers`，默认 12）、`--backend process|thread`（默认 process）和 `--overwrite`。已存在的结果默认跳过，切换输出模式时建议更换目录。

## 标签与实例

| 类别 | 含义 |
| --- | --- |
| 0 | other，其他表面 |
| 1 | hole，孔 |
| 2 | boss，凸台 |
| 3 | chamfer，倒角 |

普通 CLI 默认打印 `{"labels": [...]}`，`--verbose` 显示实例、面号和识别原因。普通模式的 `--json` 保存详细结果；批量 labels 模式保存标签列表。

full 模式输出 `[[sampleid, {"seg": {"faceid": classid}, "inst": adj_matrix}]]`。sampleid 是文件名去掉扩展名后的字符串；seg 的 JSON 键是字符串，默认从 0 起，可用 `--face-index-base 1` 改变。inst 始终按 STEP 面枚举顺序排列，为 N×N 对称矩阵：具有特征且属于同一实例的面两两为 1（包括对角线），其余为 0。它表达同实例关系，不要求两个面直接共享边。其他表面的对角线也是 0。

## 识别规则

程序构建 B-Rep 面邻接图，记录曲面类型、法向、轴线、半径、参数跨度、共享边和内环邻接。识别顺序为孔、倒角、结构化凸台，最终合并倒角实例并分配实例编号。后续规则只认领未标注面。

### 孔

从 inward 圆柱侧壁出发，沿共享边收集同轴、同半径的墙面组，检查累计 U 跨度达到完整圆周。几何共轴但拓扑不相连的墙面不会拼成孔。内环 carrier 可直接连接孔壁，也可隔一层 cone/torus 过渡面。

有 carrier 证据时，检查邻接平面与孔壁的圆弧连接是否构成完整圆周、是否位于该平面外环，据此附加盲孔底面；同轴阶梯孔通过中间 shelf 平面合并。入口 carrier 不加入孔。多边形镂空、纯平面侧壁开口不扩展为 hole。

### 倒角

支持平面倒角、锥形倒角和外凸圆柱窄条。主要依据邻接支撑面在接触位置的法向与过渡角判断；平面倒角要求近似垂直的支撑对及合适的支撑角，不使用面积上限。圆锥规则排除两端都有端板的封闭锥台，圆柱窄条规则要求外凸方向。实现还包含角帽处理、同族锥面传播及迭代识别。

围绕同一孔/凸台且彼此相邻的倒角候选按锚定实例合并；同一孔上下不相连的倒角圈保持独立。该分组依赖孔/凸台识别结果，并非任意倒角面的全局几何聚类。

### 凸台

统一检查闭合侧壁环、覆盖顶面及基准面，通过相对基准面的凸出方向区分凸台与凹槽。侧壁可由平面、圆柱和自由曲面组成；顶面允许内环，因此可表达复杂轮廓及带孔凸台。实际识别仍受种子、闭合性、法向和拓扑证据限制，并不保证任意 Logo 都能识别。

顶面需与凸出轴对齐且位于侧壁外侧，盖子边界不得泄漏到无关外表面。支持跨倒角/圆角找侧壁或顶面，也有独立圆柱、曲面 carrier、同轴法兰和轴端头部的专门判定。侧壁扫描两轮，以利用先前认领的特征作为合法边界。

凸台实例包含侧壁、顶面及完全包围的未标注内向圆柱凹陷壁；基准面、已识别孔和倒角不进入凸台。cone/torus 过渡面只作桥接，保持自身标签。不同轴向级别的圆柱侧壁不直接合并为一个环。

## 开发与验证

核心实现位于 `geometry.py`（STEP 与拓扑几何）、`recognizer.py`（识别和实例分组）；`cli.py`、`batch_predict.py`、`evaluate.py` 分别负责单文件、批量和面级评估。`scripts/` 下的抽检工具用于抽样、生成复核表及收集文件，可在该目录通过各脚本的 `--help` 查看参数。

修改识别规则后应运行数据集评估，并检查复杂凸台、孔与倒角共存、STEP 分面及误识别案例。面准确率不能替代实例级验证。阈值和具体边界以 `HintBasedRecognizer` 及其判定函数为准。
