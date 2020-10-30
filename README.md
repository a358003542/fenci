# fenci

中文分词模块：继承了jieba分词的基本算法逻辑，进行了全方位的代码优化，还额外提供了HMM算法的训练功能支持。


## 设计
### 数据存储格式
不使用marshal，这并不规范，也不使用pickle，在某些情况下确实使用pickle是必要的，但至少在这里数据格式还没必要上pickle。而是使用更通用和更安全的json数据存储格式。

模型数据就存放在 `\AppData\Local\Temp` 里面的 `fenci.cache` ，其就是一个json文件。

读写速度模型文件未建立需要1秒多，模型文件建立正常读写文件需要0.3秒多，值得一提的是本程序经过优化只要你一直调用 `s=Segment()` 同一对象，则读取模型只会读取一次，也就是后面多次cut则前面的0.3秒加载时间几乎可以忽略笔记。

## USAGE
### lcut or cut
```
from fenci.segment import Segment
segment = Segment()
res = segment.lcut("这是一段测试文字。")
```

### load_userdict
```
from fenci.segment import Segment
s = Segment()
s.load_userdict('tests/test_dict.txt')
```

### training
指定root和regexp来搜索指定文件夹下的文本，其中的文本格式如下：
```
’  我  扔  了  两颗  手榴弹  ，  他  一下子  出  溜  下去  。
```
即该分词的地方空格即可。

```
    def training(self, root=None, regexp=None):
        """
        根据已经分好词的内容来训练
        :param root:
        :param regexp:
        :return:
        """
```
注意training之后词典库还只是on-fly模式，要保存到模型需要调用方法`save_model`

### training_hmm
训练HMM模型，如果设置update_dict=True,则语料库的词语数据也会刷入进来。
```
    def training_hmm(self, root=None, regexp=None, update_dict=False):
```

### save_model
所有on-fly的词库都导入到模型里面
```
    def save_model(self, save_hmm=False):
```

### add_word
```
    def add_word(self, word, freq=1):
```
### tokenize 和 lcut
给nltk调用提供的接口

### hmm_segment
默认内部构建的hmm分词器
```
    self.hmm_segment = HMMSegment(traning_root=traning_root,
                                  traning_regexp=traning_regexp,
                                  cache_file=self.cache_file)
```
### HMMSegment
#### training
指定root和regexp来搜索指定文件夹下的文本，其中的文本格式如下：
```
’  我  扔  了  两颗  手榴弹  ，  他  一下子  出  溜  下去  。
```
即该分词的地方空格即可。

```
    def training(self, root=None, regexp=None, training_mode='update'):
```
提供了两种训练模式 update 和 replace 。

update模式将在原有HMM训练数据基础上继续训练，注意训练之后的模型数据仍是on-fly的。保存需要调用`save_model`方法。

#### save_model
将hmm_segment分词器的模型保存下来。
```
self.hmm_segment.save_model()
```

## TODO
1. 编写测试案例
2. 编写文档
3. 分词评分评估


## CHANGELOG
### 0.2.0
彻底脱离原jieba分词项目结构，整体重新设计。

### 0.1.2
加入HMM训练，重新训练数据，原字典数据较小。


