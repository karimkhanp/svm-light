// =====================================================================================
// 
//       Filename:  main.cpp
// 
//    Description:  
// 
//        Version:  1.0
//        Created:  2012年09月13日 14时08分01秒
//       Compiler:  g++
// 
//         Author:  Jie Shen
//          Email:  jieshen.sjtu@gmail.com
//
//      Institute:  APEX Data and Knowledge Management Lab
//        Address:  Shanghai Jiao Tong University
// =====================================================================================

#include "svm-light.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

int main(int argc, char* argv[])
{
    bool btrain = true;
    SVM_light svm(argv[1]);

    ifstream input("test");
    ofstream output("pred-2");
    while(input.good())
    {
        string line;
        getline(input, line, '\n');
        if(line.empty())
            break;
        stringstream ss(line);
        int label;
        ss >> label;
        string tuple;
        vector<int> idx;
        vector<float> val;
        idx.reserve(1024);
        val.reserve(1024);

        while(ss>>tuple)
        {
            if(tuple.empty())
                break;
            int _idx;
            float _val;
            sscanf(tuple.c_str(), "%d:%f", &_idx, &_val);
            idx.push_back(_idx-1);
            val.push_back(_val);
        }

        vector<float> feature(*(idx.end()-1)+1);
        for(int i=0; i <idx.size();++i)
        {
            feature[idx[i]] = val[i];
        }

        double dist = svm.classify(feature);
        output << dist << endl;
    }

    input.close();
    output.close();

    return 0;
    //if (btrain)
    {
        FILE* train_file = fopen(argv[2], "r");
        svm.train(train_file);
        fclose(train_file);
    }
    //else
    {
        FILE* test_file = fopen(argv[3], "r");
        FILE* pred_file = fopen(argv[4], "w");
        svm.classify(test_file, pred_file);
        fclose(test_file);
        fclose(pred_file);
    }
    return 0;
}
