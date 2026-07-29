## fenci

中文分词模块

本分词器采用基于词典的最大正向匹配算法为主，辅以HMM（隐马尔可夫模型）进行未登录词识别。在 SIGHAN Bakeoff 2005 数据集上的评测结果为：Precision 83.59%、Recall 84.47%、F1 84.03%，处理速度达 1145.6 KB/s。结果表明，该分词器在准确性与处理效率之间取得了良好平衡，适用于对实时性要求较高的通用文本分词场景。

### 重要提示
- 模型文件默认是 `\AppData\Local\Temp` 里面的 `fenci_model` ，该模型实际就是一个json文件。后续你可以继续训练该模型，也可以回滚该模型 （seg.reset_model()）。
- 推荐将 `seg = Segment()` 放在一个更全局的位置，而不要频繁创建它。


### 安装
```text
pip install fenci
```

### 使用
#### lcut or cut
```python
from fenci.segment import Segment
seg = Segment()
res = seg.lcut("这是一段测试文字。")
```

#### 加载自定义词库
```python
from fenci.segment import Segment
s = Segment()
s.load_userdict('tests/test_dict.txt')
```

#### 训练模型
指定root和regexp来搜索指定文件夹下的文本，其中的文本格式如下：
```
’  我  扔  了  两颗  手榴弹  ，  他  一下子  出  溜  下去  。
```
即该分词的地方空格即可。

```python
from fenci import Segment
seg = Segment()

seg.training('../icwb2-data/training', 'msr_training.utf8', with_hmm=True)

seg.save_model(save_hmm=True)
```
注意training之后词典库还只是on-fly模式，要保存到模型需要调用方法`save_model`

##### 只训练HMM模型
```python
from fenci import Segment
seg = Segment()

seg.hmm_segment.traning('../icwb2-data/training', 'msr_training.utf8')

seg.hmm_segment.save_model()
```

##### 只训练词库
```python
from fenci import Segment
seg = Segment()

seg.traning('../icwb2-data/training', 'msr_training.utf8', with_hmm=False)

seg.save_model(save_hmm=False)
```

#### 回滚模型
回滚到默认模型

```python
from fenci import Segment
s = Segment()
s.reset_model(model='default')
```


### 评估
评测使用 [SIGHAN Bakeoff 2005 金标准文件](https://github.com/yuikns/icwb2-data) ：

```text
=== 分词评测结果 ===
总词数(金标准): 106873
总词数(预测):   107996
正确词数:       90279
Precision:      83.59%
Recall:         84.47%
F1:             84.03%

=== 速度测试 ===
文本大小:       539.3 KB
重复次数:       3
平均耗时:       0.471 s
速度:           1145.6 KB/s

```