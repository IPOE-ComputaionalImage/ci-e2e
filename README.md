# viflo：计算成像系统端到端优化框架

有两种方式使用viflo：
1. 克隆本仓库，然后在viflo目录下创建、设计和分析成像系统；
2. 在单独的项目中使用viflo，则需通过pip安装viflo。

## 安装

1. （可选）创建虚拟环境
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
    如果使用Anaconda进行环境管理，可以使用以下命令（示例）创建虚拟环境：
   ```bash
   conda create -n <env_name> python=3.12
   conda activate <env_name>
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
   
3. 安装viflo
   如果直接在viflo目录下工作，则在命令行中进入viflo目录后，执行以下命令：
   ```bash
   pip install -e .
   ```
   如果在单独的项目中使用viflo，则在项目环境中安装viflo。

## 使用方法
   
### 设置参数

参考`design-files/lwir.py`，设置计算成像系统的参数。

### 训练



1. （可选）预训练神经网络，不优化光学参数

   ```bash
   viflo <design_file_path> pretrain
   ```

2. 端到端优化
   把设计文件中的`initializing_ckpt_path`设置为预训练神经网络的参数文件的路径，
   然后运行训练命令：
   ```bash
   viflo <design_file_path> train
   ```
   
### 成像性能评估

训练结果（包括参数和Tensorboard记录）保存在`tapes`目录下。
在`tapes`目录下找到训练后保存的参数文件（后缀为`.ckpt`），
将设计文件中的`trained_ckpt_path`设置为该文件的路径，然后运行测试命令：
```bash
viflo <design_file_path> eval
```
