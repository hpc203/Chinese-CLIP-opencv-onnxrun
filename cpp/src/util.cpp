#include "util.hpp"
#include <filesystem>
#include <cmath> 

std::vector<std::string> listdir(const std::string image_dir)
{
    const std::vector<std::string> exts{".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG",".heic", ".heif", ".bmp", ".webp", ".dib", ".pbm", ".pgm", ".ppm", ".emf", ".wmf", ".tiff", ".tif"};
    std::vector<std::string> imglist;
    for (const auto &entry : std::filesystem::directory_iterator(image_dir))   ///c++17标准里才有std::filesystem函数的,你也可以用C语言函数opendir来实现遍历文件夹里的全部图片,又或者使用cv::glob
    {
        if (entry.is_regular_file())
        {
            std::string ext = entry.path().extension();

            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                            { return std::tolower(c); });

            if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            {
                imglist.push_back(entry.path().string());
            }
        }
    }

    if (imglist.empty())
    {
        std::cout<<"input imgdir is empty!"<<std::endl;
    }
    return imglist;
}

void softmax(std::vector<float> &input) ///单张图片的,不考虑batchsize多个图片的
{
	const int length = input.size();
	std::vector<float> exp_x(length);
	float maxVal = *std::max_element(input.begin(), input.end());
	float sum = 0;
	for (int i = 0; i < length; i++)
	{
		const float expval = std::exp(input[i] - maxVal);
		exp_x[i] = expval;
		sum += expval;
	}
	for (int i = 0; i < length; i++)
	{
		input[i] = exp_x[i] / sum;
	}
}

int write_image_feature_name2bin(int len_feature, const float* output, const std::vector<std::string> imglist, const char* bin_name)
{
    const int imgnum = imglist.size();
    FILE* fp = fopen(bin_name, "wb");
	fwrite(&imgnum, sizeof(int), 1, fp);
	fwrite(&len_feature, sizeof(int), 1, fp);
	fwrite(output, sizeof(float), imgnum * len_feature, fp);
	for (int i = 0; i < imglist.size(); i++)   //// num_face == names.size();
	{
		int len_s = imglist[i].length();
		fwrite(&len_s, sizeof(int), 1, fp);  ///字符串的长度
		fwrite(imglist[i].c_str(), sizeof(char), len_s + 1, fp);   ///字符串末尾'\0'也算一个字符的
	}
	fclose(fp);
	return 0;
}

float* read_image_feature_name2bin(int* imgnum, int* len_feature, std::vector<std::string>& imglist, const char* bin_name)
{
	FILE* fp = fopen(bin_name, "rb");
	fread(imgnum, sizeof(int), 1, fp);
	fread(len_feature, sizeof(int), 1, fp);
	int len = (*imgnum) * (*len_feature);
	float* output = new float[len];
	fread(output, sizeof(float), len, fp);//导入数据
	for (int i = 0; i < *imgnum; i++)
	{
		int len_s = 0;
		fread(&len_s, sizeof(int), 1, fp);
		char* name = new char[len_s + 1];   ///字符串末尾'\0'也算一个字符的
		fread(name, sizeof(char), len_s + 1, fp);//导入数据
		//cout << name << endl;
		imglist.push_back(name);
		delete[] name;
        name = nullptr;
	}
	fclose(fp);//关闭文件。
	return output;
}

void copyfile_dstpath(std::vector<std::tuple<std::string, float>> imglist, std::string savepath)
{
	if (std::filesystem::exists(savepath))
    {
        std::filesystem::remove_all(savepath);
    }
	std::filesystem::create_directories(savepath);
	
	for(int i=0;i<imglist.size();i++)
	{
		std::string imgpath = std::get<0>(imglist[i]);
		////float score = get<1>(imglist[i]);
		std::filesystem::copy(imgpath, savepath);
	}
}
