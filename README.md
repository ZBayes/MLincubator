# MLincubator
The complete English version of the document will be released soon. If you wanna read, Here is a [simple version](https://github.com/ZBayes/MLincubator/blob/master/README_EN.md). 

## 一种面向实验的机器学习框架
起名字一定要霸气，技术含量不是很高，只是自己在平时做实验和建模过程中的一些经验，总结成一种建模的思路和机器学习实验的设计，方便进行多角度的实验和零件更替，同时降低开发与计算的成本，以更快地得到较好模型实现预定功能。

部分功能还未完全实现，思想已经基本提出，希望大家能够批评指正。

### 面向实验的基本需求
- 需要测试和存储多种方案
    + 既然是需要进行实验，比对方案的优劣性，那在文件中，就需要存储多个方案
- 方案中存在多个零件
    + 很多时候，方案的不同只是因为某个零件不同，如果不将零件分离，将会出现很多重复的方案，降低实验时间
- 日志系统
    + 对各种方案的结果进行记录和测算，并提供比对建议

### 基本思想
- 分块
    + 将多个步骤零件进行划分，通过记录过程结果衔接每一个步骤
- 日志
    + 记录实验结果以进行合适的对比
- 迅速包装
    + 直接调用已经存储的模型进行流程化计算

#### 分块
基本的机器学习建模流程如图所示：
![MLflow](https://github.com/ZBayes/MLincubator/raw/master/pics/pic1.JPG)
相信大家对这张图的基本结构都比较熟悉，横向我把整套机器学习建模流程分成三块，分别代表三个阶段等功能，数据探索、建立模型，模型上线三块，纵向分同样表示三块，用一切分三块大流程，数据预处理，模型建立以及计算结果三块。首先我们能够很清晰的了解一点就是，如果每一次计算都从左走到右走全流程，是非常花费时间的，再者对多个模型方案，那就需要走更多的次数，这会大大提升模型的试验运行成本，因此需要一种合适的结构进行提升。

对于多步的计算方案，一般的实验会是下面这种结构来安排实验内容和代码：
![MLflow](https://github.com/ZBayes/MLincubator/raw/master/pics/pic2.JPG)
然而，如果进行分割，将每一个算子都分开，如下图所示：
![MLflow](https://github.com/ZBayes/MLincubator/raw/master/pics/pic3.JPG)
将每一个计算的结果导出，在下一步的计算结果中调用，能很大程度降低前一步计算所需要的时间，因为此时，每一个计算都只需要执行一次，这就是分块的具体含义。

先下面是一个具体的例子：
![MLflow](https://github.com/ZBayes/MLincubator/raw/master/pics/pic4.JPG)
进行文本分类时，需要将文本转化为可以进行计算的格式，word2vector和glove都是常用的方式，此后，需要建立深度学习模型进行计算，传统的方式进行计算，则要计算4个流程，其中word2vector和glove进行了两次计算，其实对于同一套数据而言，word2vector两次计算的输出结果完全相同，不需要重复进行计算。。
![MLflow](https://github.com/ZBayes/MLincubator/raw/master/pics/pic5.JPG)
而如果进行分块，则可以降低模型的运算时间，主要是word2vector和glove不用进行两次计算，其实对于数据而言，方案1和方案2中，word2vector的输出结果完全相同，不需要进行计算。

#### 日志
日志系统仍在建设中，但是从个人经验开来，一个完善的日志，对模型的评价和选择具有很高的参考价值，同时也是进行结果展示的关键。后续还会在下面内容中完善：
1. 日志清除与清空
2. 日志格式优化

```python
import datetime
class logTool():

    path = ""
    reader = ""

    def __init__(self, path):
        self.path = path

    def writer(self, note):
        self.open()
        print(note, file=self.reader)
        self.close()

    def open(self):
        self.reader = open(self.path, "a")

    def close(self):
        self.reader.close()

    def info(self, note):
        self.open()
        print("[INFO:%s] : %s" %
              (datetime.datetime.now(), note), file=self.reader)
        self.close()

    def error(self, note):
        self.open()
        print("[ERROR:%s] : %s" %
              (datetime.datetime.now(), note), file=self.reader)
        self.close()
```

#### 迅速包装
模型做好了，需要进行全量数据实验、模型上线等工程化流程，因此需要一个快速打包的工作，我的水平有限，不会做自动化打包，但是还是可以降低打包、流程化的时间，其实实质就是将之前提到的零件进行组装，形成流程化方案。

### 具体结构
-data：数据  
|-src_data：原始数据  
|-model：存储的模型文件  
|-log：日志，存储的日志文件  
-src  
|-data_explore：数据的建缩影  
|-data_process：集中在抽样、数据规范化等过程  
|-model：一些模型方案，以及一些必要的模型零件  
|-eval：对模型结果进行分析和判断  
|-util：通用型工具函数，如日志类等  
|-flow：拼装模型，组建成完整流程  
