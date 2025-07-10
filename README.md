# 计算成像系统端到端优化框架

## 准备

### 环境搭建

1. （可选）创建虚拟环境
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
    如果使用Anaconda进行环境管理，可以使用以下命令（示例）创建虚拟环境：
   ```bash
   conda create -n e2e python=3.12
   conda activate e2e
   ```
   
2. 安装`dnois`，可以从[dnois Github仓库](https://github.com/GjQAQ/dnois.git)安装
   ```bash
   pip install git+https://github.com/GjQAQ/dnois.git
   ```
   
   或者从Github仓库下载wheel包（例如`dnois-0.1.1a1-py3-none-any.whl
`）后，在本地安装
   ```bash
   pip install dnois-0.1.1a1-py3-none-any.whl
   ```
   
3. 安装其余依赖
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
   
### 设置参数

参考`design-files/lwir.py`，设置计算成像系统的参数。

### 训练

训练的命令为
```bash
python -m e2e <design_file_path> train
```

训练结果（包括参数和Tensorboard记录）保存在`tapes`目录下。

1. （可选）预训练神经网络，不优化光学参数

   将设计文件中的`framework`设置为`'pretrain'`：
   ```python
   framework = 'pretrain'
   ```
   然后运行上述训练命令。

2. 端到端优化
   将设计文件中的`framework`设置为`'e2e'`：
   ```python
   framework = 'e2e'
   ```
   然后运行上述训练命令。
   
### 成像性能评估

在`tapes`目录下找到训练后保存的参数文件（后缀为`.ckpt`），
将设计文件中的`trained_ckpt_path`设置为该文件的路径，然后运行测试命令：
```bash
python -m e2e <design_file_path> eval
```
