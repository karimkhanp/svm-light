/*
 * svm-light.cpp
 *
 *  Created on: 2012-9-12
 *      Author: jieshen
 */

#include "svm-light.hpp"

#include "svm_common.h"
#include "svm_learn.h"

#include <stdio.h>

#include <vector>

using namespace std;

SVM_light::SVM_light(char* model_f)
{
    strcpy(modelfile, model_f);
    init();
}

SVM_light::~SVM_light()
{
    if (model != NULL)
    {
        free_model(model, 0);
        model = NULL;
    }

    free_param();
}

void SVM_light::init()
{
    strcpy(restartfile, "");
    l_parm = NULL;
    k_parm = NULL;
    model = NULL;
}

void SVM_light::free_param()
{
    if (k_parm)
    {
        free(k_parm);
        k_parm = NULL;
    }
    if (l_parm)
    {
        free(l_parm);
        l_parm = NULL;
    }
}

void SVM_light::train(FILE* train_file)
{
    verbosity = 1;

    init_param();
    MODEL* _model = (MODEL*) my_malloc(sizeof(MODEL));

    DOC **docs; /* training examples */
    long totwords, totdoc, i;
    double *target;
    double *alpha_in = NULL;
    KERNEL_CACHE *kernel_cache;

    read_documents(train_file, &docs, &target, &totwords, &totdoc);
    if (restartfile[0])
        alpha_in = read_alphas(restartfile, totdoc);

    fprintf(stderr, "Training set loaded\n");

    if (k_parm->kernel_type == LINEAR)
    {
        kernel_cache = NULL;
    }
    else
    {
        kernel_cache = kernel_cache_init(totdoc, l_parm->kernel_cache_size);
    }

    if (l_parm->type == CLASSIFICATION)
    {
        fprintf(stderr, "Start classification\n");
        svm_learn_classification(docs, target, totdoc, totwords, l_parm, k_parm,
                                 kernel_cache, _model, alpha_in);
    }
    else if (l_parm->type == REGRESSION)
    {
        svm_learn_regression(docs, target, totdoc, totwords, l_parm, k_parm,
                             &kernel_cache, _model);
    }
    else if (l_parm->type == RANKING)
    {
        svm_learn_ranking(docs, target, totdoc, totwords, l_parm, k_parm,
                          &kernel_cache, _model);
    }
    else if (l_parm->type == OPTIMIZATION)
    {
        svm_learn_optimization(docs, target, totdoc, totwords, l_parm, k_parm,
                               kernel_cache, _model, alpha_in);
    }

    if (kernel_cache)
    {
        kernel_cache_cleanup(kernel_cache);
    }

    /* Warning: The model contains references to the original data 'docs'.
     If you want to free the original data, and only keep the model, you
     have to make a deep copy of 'model'. */
    /* deep_copy_of_model=copy_model(model); */
    model = copy_model(_model);
    write_model(modelfile, model);

    free(alpha_in);
    free_model(_model, 0);
    _model = NULL;

    for (i = 0; i < totdoc; i++)
        free_example(docs[i], 1);
    free(docs);
    docs = NULL;
    free(target);
    target = NULL;

    free_param();
}

void SVM_light::classify(FILE* docfl, FILE* predfl)
{
    verbosity = 2;
    long pred_format = 1;

    DOC *doc; /* test example */
    WORD *words;
    long max_docs, max_words_doc, lld;
    long totdoc = 0, queryid, slackid;
    long correct = 0, incorrect = 0, no_accuracy = 0;
    long res_a = 0, res_b = 0, res_c = 0, res_d = 0, wnum;
    long j;
    double t1, runtime = 0;
    double dist, doc_label, costfactor;
    char *line, *comment;

    fseek(docfl, 0, SEEK_SET);
    nol_ll(docfl, &max_docs, &max_words_doc, &lld);
    max_words_doc += 2;
    lld += 2;

    line = (char *) my_malloc(sizeof(char) * lld);
    words = (WORD *) my_malloc(sizeof(WORD) * (max_words_doc + 10));

    if (model == NULL)
    {
        model = read_model(modelfile);
    }

    if (model->kernel_parm.kernel_type == 0)
    {
        add_weight_vector_to_linear_model(model);
    }

    if (verbosity >= 2)
    {
        fprintf(stderr, "Classifying test examples..");
        fflush(stderr);
    }

    while ((!feof(docfl)) && fgets(line, (int) lld, docfl))
    {
        if (line[0] == '#')
            continue;
        parse_document(line, words, &doc_label, &queryid, &slackid, &costfactor,
                       &wnum, max_words_doc, &comment);
        totdoc++;
        if (model->kernel_parm.kernel_type == 0)
        {
            for (j = 0; (words[j]).wnum != 0; j++)
            {
                if ((words[j]).wnum > model->totwords)
                    (words[j]).wnum = 0;
            }
            doc = create_example(-1, 0, 0, 0.0,
                                 create_svector(words, comment, 1.0));
            t1 = get_runtime();
            dist = classify_example_linear(model, doc);
            runtime += (get_runtime() - t1);
            free_example(doc, 1);
        }
        else
        {
            doc = create_example(-1, 0, 0, 0.0,
                                 create_svector(words, comment, 1.0));
            t1 = get_runtime();
            dist = classify_example(model, doc);
            runtime += (get_runtime() - t1);
            free_example(doc, 1);
        }
        if (dist > 0)
        {
            if (pred_format == 0)
            {
                fprintf(predfl, "%.8g:+1 %.8g:-1\n", dist, -dist);
            }
            if (doc_label > 0)
                correct++;
            else
                incorrect++;
            if (doc_label > 0)
                res_a++;
            else
                res_b++;
        }
        else
        {
            if (pred_format == 0)
            {
                fprintf(predfl, "%.8g:-1 %.8g:+1\n", -dist, dist);
            }
            if (doc_label < 0)
                correct++;
            else
                incorrect++;
            if (doc_label > 0)
                res_c++;
            else
                res_d++;
        }
        if (pred_format == 1)
        {
            fprintf(predfl, "%.8g\n", dist);
        }
        if ((int) (0.01 + (doc_label * doc_label)) != 1)
        {
            no_accuracy = 1;
        }
        if (verbosity >= 2)
        {
            if (totdoc % 100 == 0)
            {
                fprintf(stderr, "%ld..", totdoc);
                fflush(stderr);
            }
        }
    }
    free(line);
    line = NULL;
    free(words);
    words = NULL;
    free_model(model, 1);
    model = NULL;

    if (verbosity >= 2)
    {
        fprintf(stderr, "done\n");
        fprintf(stderr, "Runtime (without IO) in cpu-seconds: %.2f\n",
                (float) (runtime / 100.0));

    }
    if ((!no_accuracy) && (verbosity >= 1))
    {
        fprintf(stderr,
                "Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n",
                (float) (correct) * 100.0 / totdoc, correct, incorrect, totdoc);

        fprintf(stderr, "Precision/recall on test set: %.2f%%/%.2f%%\n",
                (float) (res_a) * 100.0 / (res_a + res_b),
                (float) (res_a) * 100.0 / (res_a + res_c));
    }

}

double SVM_light::classify(const vector<float>& feature)
{
    if (model == NULL)
    {
        model = read_model(modelfile);
    }

    //fprintf(stderr, "Model loaded\n");

    if (model->kernel_parm.kernel_type == 0)
    {
        add_weight_vector_to_linear_model(model);
    }

    WORD *words = get_word_from_feature(feature);

    DOC* doc;
    double dist(0);

    if (model->kernel_parm.kernel_type == 0)
    {
        for (int j = 0; (words[j]).wnum != 0; j++)
        {
            if ((words[j]).wnum > model->totwords)
                (words[j]).wnum = 0;
        }
        doc = create_example(-1, 0, 0, 0.0, create_svector(words, "", 1.0));
        dist = classify_example_linear(model, doc);
        free_example(doc, 1);
    }
    else
    {
        doc = create_example(-1, 0, 0, 0.0, create_svector(words, "", 1.0));
        dist = classify_example(model, doc);
        free_example(doc, 1);
    }

    free(words);
    words = NULL;

    return dist;
}

WORD* SVM_light::get_word_from_feature(const vector<float>& feature) const
{
    WORD* word = (WORD*) my_malloc(sizeof(WORD) * (feature.size() + 1));
    for (size_t i = 0; i < feature.size(); ++i)
    {
        word[i].wnum = i + 1;
        word[i].weight = feature[i];
    }
    word[feature.size()].wnum = 0;
    word[feature.size()].weight = 0;

    return word;
}

void SVM_light::init_param()
{
    l_parm = (LEARN_PARM*) my_malloc(sizeof(LEARN_PARM));
    k_parm = (KERNEL_PARM*) my_malloc(sizeof(KERNEL_PARM));

    char type[100];

    /* set default */
    strcpy(l_parm->predfile, "predictions");
    strcpy(l_parm->alphafile, "");

    l_parm->biased_hyperplane = 1;
    l_parm->sharedslack = 0;
    l_parm->remove_inconsistent = 0;
    l_parm->skip_final_opt_check = 0;
    l_parm->svm_maxqpsize = 10;
    l_parm->svm_newvarsinqp = 0;
    l_parm->svm_iter_to_shrink = -9999;
    l_parm->maxiter = 100000;
    l_parm->kernel_cache_size = 40;
    l_parm->svm_c = 0.01;
    l_parm->eps = 0.1;
    l_parm->transduction_posratio = -1.0;
    l_parm->svm_costratio = 1.0;
    l_parm->svm_costratio_unlab = 1.0;
    l_parm->svm_unlabbound = 1E-5;
    l_parm->epsilon_crit = 0.001;
    l_parm->epsilon_a = 1E-15;
    l_parm->compute_loo = 0;
    l_parm->rho = 1.0;
    l_parm->xa_depth = 0;

    k_parm->kernel_type = 0;
    k_parm->poly_degree = 3;
    k_parm->rbf_gamma = 1.0;
    k_parm->coef_lin = 1;
    k_parm->coef_const = 1;
    strcpy(k_parm->custom, "empty");
    strcpy(type, "c");

    /*
     for (i = 1; (i < argc) && ((argv[i])[0] == '-'); i++)
     {
     switch ((argv[i])[1])
     {
     case '?':
     print_help();
     exit(0);
     case 'z':
     i++;
     strcpy(type, argv[i]);
     break;
     case 'v':
     i++;
     (*_verbosity) = atol(argv[i]);
     break;
     case 'b':
     i++;
     l_parm->biased_hyperplane = atol(argv[i]);
     break;
     case 'i':
     i++;
     l_parm->remove_inconsistent = atol(argv[i]);
     break;
     case 'f':
     i++;
     l_parm->skip_final_opt_check = !atol(argv[i]);
     break;
     case 'q':
     i++;
     l_parm->svm_maxqpsize = atol(argv[i]);
     break;
     case 'n':
     i++;
     l_parm->svm_newvarsinqp = atol(argv[i]);
     break;
     case '#':
     i++;
     l_parm->maxiter = atol(argv[i]);
     break;
     case 'h':
     i++;
     l_parm->svm_iter_to_shrink = atol(argv[i]);
     break;
     case 'm':
     i++;
     l_parm->kernel_cache_size = atol(argv[i]);
     break;
     case 'c':
     i++;
     l_parm->svm_c = atof(argv[i]);
     break;
     case 'w':
     i++;
     l_parm->eps = atof(argv[i]);
     break;
     case 'p':
     i++;
     l_parm->transduction_posratio = atof(argv[i]);
     break;
     case 'j':
     i++;
     l_parm->svm_costratio = atof(argv[i]);
     break;
     case 'e':
     i++;
     l_parm->epsilon_crit = atof(argv[i]);
     break;
     case 'o':
     i++;
     l_parm->rho = atof(argv[i]);
     break;
     case 'k':
     i++;
     l_parm->xa_depth = atol(argv[i]);
     break;
     case 'x':
     i++;
     l_parm->compute_loo = atol(argv[i]);
     break;
     case 't':
     i++;
     k_parm->kernel_type = atol(argv[i]);
     break;
     case 'd':
     i++;
     k_parm->poly_degree = atol(argv[i]);
     break;
     case 'g':
     i++;
     k_parm->rbf_gamma = atof(argv[i]);
     break;
     case 's':
     i++;
     k_parm->coef_lin = atof(argv[i]);
     break;
     case 'r':
     i++;
     k_parm->coef_const = atof(argv[i]);
     break;
     case 'u':
     i++;
     strcpy(k_parm->custom, argv[i]);
     break;
     case 'l':
     i++;
     strcpy(l_parm->predfile, argv[i]);
     break;
     case 'a':
     i++;
     strcpy(l_parm->alphafile, argv[i]);
     break;
     case 'y':
     i++;
     strcpy(restartfile, argv[i]);
     break;
     default:
     fprintf(stderr,"\nUnrecognized option %s!\n\n", argv[i]);
     print_help();
     exit(0);
     }
     }*/

    if (l_parm->svm_iter_to_shrink == -9999)
    {
        if (k_parm->kernel_type == LINEAR)
        {
            l_parm->svm_iter_to_shrink = 2;
        }
        else
            l_parm->svm_iter_to_shrink = 100;
    }
    if (strcmp(type, "c") == 0)
    {
        l_parm->type = CLASSIFICATION;
    }
    else if (strcmp(type, "r") == 0)
    {
        l_parm->type = REGRESSION;
    }
    else if (strcmp(type, "p") == 0)
    {
        l_parm->type = RANKING;
    }
    else if (strcmp(type, "o") == 0)
    {
        l_parm->type = OPTIMIZATION;
    }
    else if (strcmp(type, "s") == 0)
    {
        l_parm->type = OPTIMIZATION;
        l_parm->sharedslack = 1;
    }
    else
    {
        fprintf(stderr,
                "\nUnknown type '%s': Valid types are 'c' (classification), 'r' regession, and 'p' preference ranking.\n",
                type);
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if ((l_parm->skip_final_opt_check) && (k_parm->kernel_type == LINEAR))
    {
        fprintf(stderr,
                "\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
        l_parm->skip_final_opt_check = 0;
    }
    if ((l_parm->skip_final_opt_check) && (l_parm->remove_inconsistent))
    {
        fprintf(stderr,
                "\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if ((l_parm->svm_maxqpsize < 2))
    {
        fprintf(stderr,
                "\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",
                l_parm->svm_maxqpsize);
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if ((l_parm->svm_maxqpsize < l_parm->svm_newvarsinqp))
    {
        fprintf(stderr,
                "\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",
                l_parm->svm_maxqpsize);

        fprintf(stderr,
                "new variables [%ld] entering the working set in each iteration.\n",
                l_parm->svm_newvarsinqp);

        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if (l_parm->svm_iter_to_shrink < 1)
    {

        fprintf(stderr,
                "\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",
                l_parm->svm_iter_to_shrink);
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if (l_parm->svm_c < 0)
    {
        fprintf(stderr, "\nThe C parameter must be greater than zero!\n\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if (l_parm->transduction_posratio > 1)
    {
        fprintf(stderr,
                "\nThe fraction of unlabeled examples to classify as positives must\n");
        fprintf(stderr, "be less than 1.0 !!!\n\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if (l_parm->svm_costratio <= 0)
    {
        fprintf(stderr,
                "\nThe COSTRATIO parameter must be greater than zero!\n\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if (l_parm->epsilon_crit <= 0)
    {
        fprintf(stderr,
                "\nThe epsilon parameter must be greater than zero!\n\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if (l_parm->rho < 0)
    {
        fprintf(stderr,
                "\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
        fprintf(stderr,
                "be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
        fprintf(stderr,
                "Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
    if ((l_parm->xa_depth < 0) || (l_parm->xa_depth > 100))
    {
        fprintf(stderr,
                "\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
        fprintf(stderr,
                "for switching to the conventional xa/estimates described in T. Joachims,\n");
        fprintf(stderr,
                "Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
        wait_any_key();
        print_help();
        exit(EXIT_FAILURE);
    }
}

void SVM_light::wait_any_key() const
{
    printf("\n(more)\n");
    (void) getc(stdin);
}

void SVM_light::print_help() const
{
    fprintf(stderr,
            "\nSVM-light %s: Support Vector Machine, learning module     %s\n",
            VERSION, VERSION_DATE);
    copyright_notice();
    fprintf(stderr,
            "   usage: svm_learn [options] example_file model_file\n\n");
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "         example_file-> file with training data\n");
    fprintf(stderr,
            "         model_file  -> file to store learned decision rule in\n");

    fprintf(stderr, "General options:\n");
    fprintf(stderr, "         -?          -> this help\n");
    fprintf(stderr, "         -v [0..3]   -> verbosity level (default 1)\n");
    fprintf(stderr, "Learning options:\n");
    fprintf(stderr,
            "         -z {c,r,p}  -> select between classification (c), regression (r),\n");
    fprintf(stderr,
            "                        and preference ranking (p) (default classification)\n");
    fprintf(stderr,
            "         -c float    -> C: trade-off between training error\n");
    fprintf(stderr,
            "                        and margin (default [avg. x*x]^-1)\n");
    fprintf(stderr,
            "         -w [0..]    -> epsilon width of tube for regression\n");
    fprintf(stderr, "                        (default 0.1)\n");
    fprintf(stderr,
            "         -j float    -> Cost: cost-factor, by which training errors on\n");
    fprintf(stderr,
            "                        positive examples outweight errors on negative\n");
    fprintf(stderr, "                        examples (default 1) (see [4])\n");
    fprintf(stderr,
            "         -b [0,1]    -> use biased hyperplane (i.e. x*w+b>0) instead\n");
    fprintf(stderr,
            "                        of unbiased hyperplane (i.e. x*w>0) (default 1)\n");
    fprintf(stderr,
            "         -i [0,1]    -> remove inconsistent training examples\n");
    fprintf(stderr, "                        and retrain (default 0)\n");
    fprintf(stderr, "Performance estimation options:\n");
    fprintf(stderr,
            "         -x [0,1]    -> compute leave-one-out estimates (default 0)\n");
    fprintf(stderr, "                        (see [5])\n");
    fprintf(stderr,
            "         -o ]0..2]   -> value of rho for XiAlpha-estimator and for pruning\n");
    fprintf(stderr,
            "                        leave-one-out computation (default 1.0) (see [2])\n");
    fprintf(stderr,
            "         -k [0..100] -> search depth for extended XiAlpha-estimator \n");
    fprintf(stderr, "                        (default 0)\n");
    fprintf(stderr, "Transduction options (see [3]):\n");
    fprintf(stderr,
            "         -p [0..1]   -> fraction of unlabeled examples to be classified\n");
    fprintf(stderr,
            "                        into the positive class (default is the ratio of\n");
    fprintf(stderr,
            "                        positive and negative examples in the training data)\n");
    fprintf(stderr, "Kernel options:\n");
    fprintf(stderr, "         -t int      -> type of kernel function:\n");
    fprintf(stderr, "                        0: linear (default)\n");
    fprintf(stderr, "                        1: polynomial (s a*b+c)^d\n");
    fprintf(stderr,
            "                        2: radial basis function exp(-gamma ||a-b||^2)\n");
    fprintf(stderr, "                        3: sigmoid tanh(s a*b + c)\n");
    fprintf(stderr,
            "                        4: user defined kernel from kernel.h\n");
    fprintf(stderr,
            "         -d int      -> parameter d in polynomial kernel\n");
    fprintf(stderr, "         -g float    -> parameter gamma in rbf kernel\n");
    fprintf(stderr,
            "         -s float    -> parameter s in sigmoid/poly kernel\n");
    fprintf(stderr,
            "         -r float    -> parameter c in sigmoid/poly kernel\n");
    fprintf(stderr,
            "         -u string   -> parameter of user defined kernel\n");
    fprintf(stderr, "Optimization options (see [1]):\n");
    fprintf(stderr,
            "         -q [2..]    -> maximum size of QP-subproblems (default 10)\n");
    fprintf(stderr,
            "         -n [2..q]   -> number of new variables entering the working set\n");
    fprintf(stderr,
            "                        in each iteration (default n = q). Set n<q to prevent\n");
    fprintf(stderr, "                        zig-zagging.\n");
    fprintf(stderr,
            "         -m [5..]    -> size of cache for kernel evaluations in MB (default 40)\n");
    fprintf(stderr, "                        The larger the faster...\n");
    fprintf(stderr,
            "         -e float    -> eps: Allow that error for termination criterion\n");
    fprintf(stderr,
            "                        [y [w*x+b] - 1] >= eps (default 0.001)\n");
    fprintf(stderr,
            "         -y [0,1]    -> restart the optimization from alpha values in file\n");
    fprintf(stderr,
            "                        specified by -a option. (default 0)\n");
    fprintf(stderr,
            "         -h [5..]    -> number of iterations a variable needs to be\n");
    fprintf(stderr,
            "                        optimal before considered for shrinking (default 100)\n");
    fprintf(stderr,
            "         -f [0,1]    -> do final optimality check for variables removed\n");
    fprintf(stderr,
            "                        by shrinking. Although this test is usually \n");
    fprintf(stderr,
            "                        positive, there is no guarantee that the optimum\n");
    fprintf(stderr,
            "                        was found if the test is omitted. (default 1)\n");
    fprintf(stderr,
            "         -y string   -> if option is given, reads alphas from file with given\n");
    fprintf(stderr,
            "                        and uses them as starting point. (default 'disabled')\n");
    fprintf(stderr,
            "         -# int      -> terminate optimization, if no progress after this\n");
    fprintf(stderr,
            "                        number of iterations. (default 100000)\n");
    fprintf(stderr, "Output options:\n");
    fprintf(stderr,
            "         -l string   -> file to write predicted labels of unlabeled\n");
    fprintf(stderr,
            "                        examples into after transductive learning\n");
    fprintf(stderr,
            "         -a string   -> write all alphas to this file after learning\n");
    fprintf(stderr,
            "                        (in the same order as in the training set)\n");
    wait_any_key();
    fprintf(stderr, "\nMore details in:\n");
    fprintf(stderr,
            "[1] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in\n");
    fprintf(stderr,
            "    Kernel Methods - Support Vector Learning, B. Sch�lkopf and C. Burges and\n");
    fprintf(stderr, "    A. Smola (ed.), MIT Press, 1999.\n");
    fprintf(stderr,
            "[2] T. Joachims, Estimating the Generalization performance of an SVM\n");
    fprintf(stderr,
            "    Efficiently. International Conference on Machine Learning (ICML), 2000.\n");
    fprintf(stderr,
            "[3] T. Joachims, Transductive Inference for Text Classification using Support\n");
    fprintf(stderr,
            "    Vector Machines. International Conference on Machine Learning (ICML),\n");
    fprintf(stderr, "    1999.\n");
    fprintf(stderr,
            "[4] K. Morik, P. Brockhausen, and T. Joachims, Combining statistical learning\n");
    fprintf(stderr,
            "    with a knowledge-based approach - A case study in intensive care  \n");
    fprintf(stderr,
            "    monitoring. International Conference on Machine Learning (ICML), 1999.\n");
    fprintf(stderr,
            "[5] T. Joachims, Learning to Classify Text Using Support Vector\n");
    fprintf(stderr,
            "    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,\n");
    fprintf(stderr, "    2002.\n\n");
}
