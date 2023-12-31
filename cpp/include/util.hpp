#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

std::vector<std::string> listdir(const std::string image_dir);
void softmax(std::vector<float> &input); ///单张图片的,不考虑batchsize多个图片的

int write_image_feature_name2bin(int len_feature, const float* output, const std::vector<std::string> imglist, const char* bin_name);
float* read_image_feature_name2bin(int* imgnum, int* len_feature, std::vector<std::string>& imglist, const char* bin_name);

void copyfile_dstpath(std::vector<std::tuple<std::string, float>> imglist, std::string savepath);

// 实现argsort功能 ,模板定义通常写在头文件里
template<typename T> std::vector<int> argsort_ascend(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) {return (array[pos1] < array[pos2]); });

    return array_index;
}

#endif