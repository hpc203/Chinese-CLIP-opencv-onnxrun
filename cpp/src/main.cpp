#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "Tokenizer.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;
using namespace dnn;
using namespace Ort;

typedef struct
{
	string name;
	float prob;
} class_info;

class Clip
{
public:
	Clip(string image_modelpath, string text_modelpath, string vocab_path);
	void generate_image_feature(Mat cv_image);
	void generate_text_feature(std::vector<std::string> texts);
	class_info zero_shot_image_classify(Mat cv_image, std::vector<std::string> texts);
	void generate_imagedir_features(const std::string image_dir, const char* bin_name);
	std::vector<std::tuple<std::string, float>> input_text_search_image(std::string text, const float* image_features, const std::vector<std::string> imglist);

private:
	Net net;  ////image_model
	Mat normalize_(Mat img);
	const int inpWidth = 224;
	const int inpHeight = 224;
	float mean[3] = { 0.48145466, 0.4578275, 0.40821073 };
	float std[3] = { 0.26862954, 0.26130258, 0.27577711 };

	std::shared_ptr<TokenizerBase> tokenizer;
	std::vector<float> image_features_input;
	std::vector<vector<float>> text_features_input;
	std::vector<int> text_tokens_input;
	bool load_tokenizer(std::string vocab_path);
	
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "CLIP_text_model");
	Ort::Session *ort_session = nullptr;  ////text_model
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
	const int context_length = 52;
	const int len_text_feature = 512;
};

Clip::Clip(string image_modelpath, string text_modelpath, string vocab_path)
{
	this->net = readNet(image_modelpath); ///opencv4.5��ȡ������.opencv4.7�Ϳ��Լ��سɹ�

	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, text_modelpath.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	////context_length = input_node_dims[0][1];
	///len_text_feature = output_node_dims[0][1];
	this->load_tokenizer(vocab_path);
}

bool Clip::load_tokenizer(std::string vocab_path)
{
	tokenizer.reset(new TokenizerClipChinese);
	this->text_tokens_input = std::vector<int>(1024 * this->context_length);
	return tokenizer->load_tokenize(vocab_path);
}

Mat Clip::normalize_(Mat img)
{
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB);
	vector<cv::Mat> rgbChannels(3);
	split(rgbimg, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* std[c]), (0.0 - mean[c]) / std[c]);
	}
	Mat m_normalized_mat;
	merge(rgbChannels, m_normalized_mat);
	return m_normalized_mat;
}

void Clip::generate_image_feature(Mat srcimg)
{
	Mat temp_image;
	resize(srcimg, temp_image, cv::Size(this->inpWidth, this->inpHeight), 0, 0, INTER_CUBIC);
	Mat normalized_mat = this->normalize_(temp_image);
	Mat blob = blobFromImage(normalized_mat);
	this->net.setInput(blob);
	vector<Mat> outs;
	////net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	float* ptr_feat = (float*)outs[0].data;
	const int len_image_feature = outs[0].size[1];  ///忽律第0维batchsize=1, len_image_feature是定值512,跟len_text_feature相等的, 也可以写死在类成员变量里
	this->image_features_input.resize(len_image_feature);
	float norm = 0.0;
	for (int i = 0; i < len_image_feature; i++)
	{
		norm += ptr_feat[i] * ptr_feat[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < len_image_feature; i++)
	{
		this->image_features_input[i] = ptr_feat[i] / norm;
	}
}

void Clip::generate_text_feature(std::vector<std::string> texts)
{
	std::vector<std::vector<int>> text_token;
	text_token.resize(texts.size());
	for (size_t i = 0; i < texts.size(); i++)
	{
		this->tokenizer->encode_text(texts[i], text_token[i]);
	}
	
	if (text_token.size() * this->context_length > text_tokens_input.size())
	{
		text_tokens_input.resize(text_token.size() * this->context_length);
	}

	memset(text_tokens_input.data(), 0, text_token.size() * this->context_length * sizeof(int));
	auto text_tokens_input_ptr = text_tokens_input.data();
	for (size_t i = 0; i < text_token.size(); i++)
	{
		if (text_token[i].size() > this->context_length)
		{
			printf("text_features index %ld ,bigger than %d\n", i, this->context_length);
			continue;
		}
		memcpy(text_tokens_input_ptr + i * this->context_length, text_token[i].data(), text_token[i].size() * sizeof(int));
	}
	
	std::vector<int64_t> text_token_shape = { 1, this->context_length };
	this->text_features_input.resize(text_token.size());

	std::vector<int64> text_tokens_input_64(texts.size() * this->context_length);
	for (size_t i = 0; i < text_tokens_input_64.size(); i++)
	{
		text_tokens_input_64[i] = text_tokens_input[i];
	}

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	for (size_t i = 0; i < text_token.size(); i++)
	{
		auto inputTensor = Ort::Value::CreateTensor<int64>(allocator_info, text_tokens_input_64.data() + i * this->context_length, this->context_length, text_token_shape.data(), text_token_shape.size());

		Ort::RunOptions runOptions;
		vector<Value> ort_outputs = ort_session->Run(runOptions, &input_names[0], &inputTensor, 1, output_names.data(), output_names.size());
		const float *text_feature_ptr = ort_outputs[0].GetTensorMutableData<float>();
		
		this->text_features_input[i].resize(this->len_text_feature);
		float norm = 0.0;
		for (int j = 0; j < this->len_text_feature; j++)
		{
			norm += text_feature_ptr[j] * text_feature_ptr[j];
		}
		norm = sqrt(norm);
		for (int j = 0; j < this->len_text_feature; j++)
		{
			this->text_features_input[i][j] = text_feature_ptr[j] / norm;
		}

	}
}

class_info Clip::zero_shot_image_classify(Mat cv_image, std::vector<std::string> texts)
{
	this->generate_image_feature(cv_image);
	this->generate_text_feature(texts);
	vector<float> logits_per_image(texts.size());  ///单张图片的,不考虑batchsize多个图片的
	for (int i = 0; i < this->text_features_input.size(); i++)
	{
		float sum = 0;
		for (int j = 0; j < len_text_feature; j++)
		{
			sum += this->image_features_input[j] * this->text_features_input[i][j]; ////图片特征向量跟文本特征向量做内积
		}
		logits_per_image[i] = 100 * sum;
	}
	softmax(logits_per_image);
	int maxPosition = std::max_element(logits_per_image.begin(), logits_per_image.end()) - logits_per_image.begin(); ///最大值的下标
	class_info result = { texts[maxPosition], logits_per_image[maxPosition] };
	return result;
}

void Clip::generate_imagedir_features(const std::string image_dir, const char* bin_name)
{
	std::vector<std::string> imglist = listdir(image_dir);
	const int imgnum = imglist.size();
	cout<<"遍历到"<<imgnum<<"张图片"<<endl;
	float* imagedir_features = new float[imgnum*this->len_text_feature];
	for(int i=0;i<imgnum;i++)
	{
		string imgpath = imglist[i];
		Mat srcimg = imread(imgpath);
		this->generate_image_feature(srcimg);
		memcpy(imagedir_features + i * len_text_feature, this->image_features_input.data(), len_text_feature * sizeof(float));
	}
	
	write_image_feature_name2bin(this->len_text_feature, imagedir_features, imglist, bin_name);

	delete [] imagedir_features;
	imagedir_features = nullptr;
}

std::vector<std::tuple<std::string, float>> Clip::input_text_search_image(std::string text, const float* image_features, const std::vector<std::string> imglist)
{
	const int imgnum = imglist.size();
	std::vector<std::string> texts = {text};
	this->generate_text_feature(texts);
	vector<float> logits_per_image(imgnum);  
	for (int i = 0; i < imgnum; i++)
	{
		float sum = 0;
		for (int j = 0; j < len_text_feature; j++)
		{
			sum += image_features[i*len_text_feature+j] * this->text_features_input[0][j]; ////图片特征向量跟文本特征向量做内积
		}
		logits_per_image[i] = 100 * sum;
	}
	softmax(logits_per_image);
	std::vector<int> index = argsort_ascend(logits_per_image); ///注意此处是从小到大排列的
	std::vector<std::tuple<std::string, float>> top5imglist(5);
	for(int i=0;i<top5imglist.size();i++)
	{
		const int ind = index[imgnum-1 - i];
		std::tuple<std::string, float> result = std::make_tuple(imglist[ind], logits_per_image[ind]);
		top5imglist[i] = result;
	}
	return top5imglist;
}


int main()
{
	Clip mynet("/project/chinese-clip-cpp/image_model.onnx", "/project/chinese-clip-cpp/text_model.onnx", "/project/chinese-clip-cpp/myvocab.txt");

	const std::string image_dir = "/project/chinese-clip-cpp/testimgs";
	const char* bin_name = "image_features.bin";

	///第一步,输入文件夹，生成图片的特征向量，保存到数据库文件
	///mynet.generate_imagedir_features(image_dir, bin_name);

	///第二步,输入一句话, 计算最相似的图片
	string input_text = "踢足球的人";
	std::string savepath = "/project/chinese-clip-cpp/resultimgs";

	int imgnum = 0, len_feature = 0;
	vector<string> imglist;
	float* imagedir_features = read_image_feature_name2bin(&imgnum, &len_feature, imglist, bin_name);
	printf("读取 %s 成功\n", bin_name);
	cout<<"有"<<imgnum<<"张图片 , 特征向量长度="<<len_feature<<endl;

	std::vector<std::tuple<std::string, float>> top5imglist = mynet.input_text_search_image(input_text, imagedir_features, imglist);

	copyfile_dstpath(top5imglist, savepath);

	delete [] imagedir_features;
	imagedir_features = nullptr;


	//////输入提示词, 做图片分类
	/*string imgpath = "/project/chinese-clip-cpp/pokemon.jpeg";
	std::vector<std::string> texts = { "杰尼龟", "妙蛙种子", "小火龙", "皮卡丘" };
	Mat srcimg = imread(imgpath);
	class_info result = mynet.zero_shot_image_classify(srcimg, texts);
	cout << "最大概率：" << result.prob << ", 对应类别：" << result.name << endl;*/
	
	return 0;
}