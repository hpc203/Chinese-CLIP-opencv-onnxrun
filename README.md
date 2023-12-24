OpenAI发布的Clip是一个基于图像和文本并行的多模态模型，CLIP模型的效果实现了图像和文本向同一个特征空间映射的能力。
当进行图像识别时，我们将待识别的图像映射成一个特征向量。同时我们将所有的类别文本转换成一个句子，然后将这个句子映射成另外一组特征向量。

文本特征向量和图像特征向量最相近的那一个便是我们要识别的目标图像的类。
Clip的玩法有很多种：
(1).Zero Shot 分类
将图像与类别文本相比较，以得到最接近的分类结果。
(2).Retrieval
对于给定的输入文本和图片相册，在文本与图片相册中找到最匹配的结果。
(3).CLIP Guidance
对于图像生成模型，结合生成结果与输入文本间的 CLIP 距离生成更好的图片。例如，扩散模型里面就用到了clip。
(4).CLIP Captioning
利用 CLIP encoding 提取特征，以该特征的映射结果结合 GPT2 生成 caption。
(5).搜索相似图片

这里面我最感兴趣的是第二项：给出一句话来描述想要的图片，就能从图库中搜出来符合要求的。现在市场上已经有这个功能的软件了，例如这里
https://github.com/mazzzystar/Queryable


在github上clip部署的程序，文本都是英文的，于是我就想着编写一套中文clip的图文检索程序。
图像模块和文本模块的onnx文件在百度云盘，链接：https://pan.baidu.com/s/18eBA19kMqdJpP5muV9V18w 
提取码：d30y
程序很简陋，感兴趣的开发者可以添加一个图形界面，显示输入文字和图库中搜出来符合要求的图片，这样看起来更直观。

目前这套程序我只编写了python的，c++的暂时没有编写。原因是在这套程序里有一个模块是clip_tokenizer，这是一个NLP领域的模块，
而我一直是做图像视觉的，不了解clip_tokenizer这个模块的细节。而我在github上看到的clip_tokenizer的c++实现的代码，是针对英文的。
中文的clip_tokenizer的c++实现的代码，有一个仓库里有 https://github.com/ozanarmagan/clip_tokenizer_cpp/blob/master/clip_tokenizer.cpp
，里面有个is_chinese_char函数，判断字符是不是中文的。但是我下载程序之后，编译出错了。
