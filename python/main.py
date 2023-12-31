import cv2
import onnxruntime as ort
import numpy as np
import os
import shutil
import pickle
from clip_tokenizer import tokenize

class Clip():
    def __init__(self, image_modelpath, text_modelpath):
        self.img_model = cv2.dnn.readNet(image_modelpath)
        self.input_height, self.input_width = 224, 224
        
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073],
                             dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.26862954, 0.26130258, 0.27577711],
                            dtype=np.float32).reshape((1, 1, 3))
        
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.txt_model = ort.InferenceSession(text_modelpath, so)
        self.context_length = 52

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height),
                         interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32)/255.0 - self.mean) / self.std
        return img

    def generate_image_feature(self, srcimg):
        img = self.preprocess(srcimg)
        blob = cv2.dnn.blobFromImage(img)
        self.img_model.setInput(blob)
        image_features = self.img_model.forward(self.img_model.getUnconnectedOutLayersNames())[0]

        img_norm = np.linalg.norm(image_features, axis=-1, keepdims=True)
        image_features /= img_norm
        return image_features
    
    def generate_text_feature(self, input_text):
        text = tokenize(input_text, context_length=self.context_length) 
        text_features = []
        for i in range(len(text)):
            one_text = np.expand_dims(text[i],axis=0)
            text_feature = self.txt_model.run(None, {self.txt_model.get_inputs()[0].name: one_text})[0].squeeze()
            text_features.append(text_feature)
        text_features = np.stack(text_features, axis=0)
        txt_norm = np.linalg.norm(text_features, axis=1, keepdims=True)
        text_features /= txt_norm
        return text_features
    
    def run_image_classify(self, image, input_strs):
        image_features = self.generate_image_feature(image)
        text_features = self.generate_text_feature(input_strs)
        logits_per_image = 100 * np.dot(image_features, text_features.T)
        exp_logits = np.exp(logits_per_image - np.max(logits_per_image, axis=-1, keepdims=True))  ###softmax
        softmax_logit = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)   ###softmax
        max_str = input_strs[softmax_logit.argmax()]
        max_str_logit = softmax_logit.max()
        return max_str, max_str_logit
    
    def generate_imagedir_features(self, image_dir):
        imglist, image_features = [], []
        for imgname in os.listdir(image_dir):
            srcimg = cv2.imread(os.path.join(image_dir, imgname))
            if srcimg is None:  ###有可能当前文件不是图片
                continue
            img_feat = self.generate_image_feature(srcimg)
            image_features.append(img_feat.squeeze())
            imglist.append(imgname)
        
        image_features = np.stack(image_features, axis=0)
        return image_features, imglist
    
    def input_text_search_image(self, input_text, image_features, imglist):
        text_features = self.generate_text_feature(input_text)
        logits_per_image = 100 * np.dot(text_features, image_features.T)
        exp_logits = np.exp(logits_per_image - np.max(logits_per_image, axis=-1, keepdims=True))  ###softmax
        softmax_logit = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)   ###softmax
        softmax_logit = softmax_logit.reshape(-1)  ### 拉平数组
        similar_id = np.argsort(-softmax_logit)  ### 降序排列
        top5_imglist = [(imglist[similar_id[i]], softmax_logit[similar_id[i]]) for i in range(5)]
        return top5_imglist

            
if __name__=='__main__':
    mynet = Clip("image_model.onnx", "text_model.onnx")

    ###第一步,输入文件夹，生成图片的特征向量，保存到数据库文件
    # image_dir = os.path.join(os.getcwd(), 'testimgs')
    # image_features, imglist = mynet.generate_imagedir_features(image_dir)
    # with open('features.pkl', 'wb') as f:
    #     pickle.dump((image_features, imglist), f)  ###文件夹确定之后, 图片特征向量计算出来之后就是一个定值了,后面第二步的时候, 就加载它, 因为每次输入的文字可能不一样,这时不需要再重复计算图片特征向量了
    # print('生成特征向量数据库成功!!!')

    ###第二步,输入一句话, 计算最相似的图片
    input_text = "踢足球的人"   ####第一步生成了特征向量数据库,这时候每当输入新的文本时,就不需要再重新计算图片特征向量
    with open('features.pkl', 'rb') as f:
        image_features, imglist = pickle.load(f)
    top5_imglist = mynet.input_text_search_image(input_text, image_features, imglist)
    print(top5_imglist)
    
    image_dir = os.path.join(os.getcwd(), 'testimgs')
    result_imgs = os.path.join(os.getcwd(), 'result_imgs')
    if os.path.exists(result_imgs):
        shutil.rmtree(result_imgs)
    os.makedirs(result_imgs)
    for imgname,conf in top5_imglist:
        shutil.copy(os.path.join(image_dir, imgname), result_imgs)


    #####输入提示词, 做图片分类
    # imgpath = 'pokemon.jpeg'
    # text = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

    # mynet = Clip("image_model.onnx", "text_model.onnx")

    # srcimg = cv2.imread(imgpath)
    # max_str, max_str_logit = mynet.run_image_classify(srcimg, text)
    # print(f"最大概率：{max_str_logit}, 对应类别：{max_str}")