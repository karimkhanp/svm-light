/*
 * svm-light.hpp
 *
 *  Created on: 2012-9-12
 *      Author: jieshen
 */

#ifndef __COUPLING_SVM_LIGHT_HPP__
#define __COUPLING_SVM_LIGHT_HPP__

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstdarg>
#include <string.h>
#include <vector>
#include <string>

class LEARN_PARM;
class KERNEL_PARM;
class MODEL;
class WORD;

using namespace std;

class SVM_light
{
public:
    explicit SVM_light(char* model_file);
    ~SVM_light();
public:
    void train(FILE* train_file);
    void classify(FILE* test_file, FILE* pred_file);
    double classify(const vector<float>& feature);
    void print_help() const;
private:
    void init();
    void init_param();
    void free_param();
    void wait_any_key() const;
    WORD* get_word_from_feature(const vector<float>& feature) const;

private:
    LEARN_PARM* l_parm;
    KERNEL_PARM* k_parm;
    MODEL* model;

    char modelfile[200]; /* file for resulting classifier */
    char restartfile[200]; /* file with initial alphas */
};

#endif /* __COUPLING_SVM_LIGHT_HPP__ */
