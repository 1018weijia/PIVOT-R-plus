# 复现过程记录

## 安装相关依赖
- 创建环境并安装相关依赖包，记住Python版本需要>=3.9.2，不然会有奇怪的问题出现
```
pip install -r requirements.txt
```
- 下载 [simulator](https://drive.google.com/file/d/1GRe5OFmQdMJIIs8EU7kobWoCyFVfMHct/view?usp=sharing)
- 下载 [checkpoint](https://drive.google.com/file/d/12uDk9m4vxqkoZCd_7vNsz_NXhaRY49t7/view?usp=drive_link)然后放到"runs"文件夹中.
- 下载 [test dataset](https://drive.google.com/file/d/15V82C2RCyfEfJKA_HYomf-YuTaiG-4gK/view?usp=sharing)放到"datasets"文件夹中. 需要训练则同样下载[train dataset](https://pan.baidu.com/s/1u5u7-gS2RjUBprMArAyBZA?pwd=7mo4 )

## 下载相应的模型权重文件
- 下载[llava-v1.5-13b模型](https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main)，**需要注意放置的路径**。在`LLAVA`文件夹下，创建文件夹`liuhaotian`，在下面再创建文件夹`llava-v1.5-13b`，将文件都放在这个文件夹下。例如，其中的`config.json`的文件路径需要为`LLAVA/liuhaotian/llava-v1.5-13b/config.json`
- 下载[clip-vit-large-patch14-336模型](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main)，**需要注意放置的路径**。同样地，在`LLAVA`文件夹下，创建文件夹`openai`，在下面再创建文件夹`clip-vit-large-patch14-336`，将文件都放在这个文件夹下。例如，其中的`config.json`的文件路径需要为`LLaVA/openai/clip-vit-large-patch14-336/config.json`
- 下载[CLIP-ViT-B-32-laion2B-s34B-b79K模型](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/tree/main)，**需要注意放置的路径**。这个路径不需要在`LLAVA`之下，在`PIVOT-R`下创建文件夹`laion`，与`LLAVA`是在同一级下，再在其下创建文件夹`CLIP-ViT-B-32-laion2B-s34B-b79K`，将所有文件都放在这个文件夹下。例如，其中`config.json`文件的路径需要为`PIVOT-R/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/config.json`

## 修改yaml文件
- 修改`PIVOT-R`下的`config`文件下的两个`yaml`文件，将`trainer.yaml`和`model/default.yaml`文件中所有`clip_path`的内容都换成`laion/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin`文件所在的绝对路径

## 修改数据集路径
- 修改上面的`yaml`文件中的`env/eval/data_path`的路径为实际的第一步中下载的数据集的路径


## 运行
- 首先打开仿真环境
    ```
    cd /path/to/simulator
    ./HARIX_RDKSim.sh -graphicsadapter=0 -port=30007 -RenderOffScreen
    ```
- 新开一个终端，部署LLAVA模型
    ```
    cd LLAVA
    python -m llava.serve.api --model-path liuhaotian/llava-v1.5-13b --temperature 0.0
    ```
- 再开一个终端，运行测试代码
    ```
    python src/tester.py
    ```

## 结果
- 最后会在PIVOT文件夹下的test文件夹中找到输出的视频