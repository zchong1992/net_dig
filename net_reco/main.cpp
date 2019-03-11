/*
* MIT License
*
* Copyright (c) 2017 wen.gu <454727014@qq.com>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <iostream>
#include <time.h>

#include "bp_neuron_net.h"

#include "data_input.h"
#include "neuron_utils.h"

/** indicate 0 ~ 9 */
#define NUM_NET_OUT 10
#define NUM_HIDDEN 200
#define NET_LEARNING_RATE 0.4

#define TRAIN_IMAGES_URL "../docs/train-images.idx3-ubyte"
#define TRAIN_LABELS_URL "../docs/train-labels.idx1-ubyte"

#define TEST_IMANGES_URL "../docs/t10k-images.idx3-ubyte"
#define TEST_LABELS_URL  "../docs/t10k-labels.idx1-ubyte"

typedef std::vector<int> InputIndex;


void showNumber(unsigned char pic[], int width, int height)
{
    int idx = 0;
    for (int i=0; i < height; i++)
    {
        for (int j = 0; j < width; j++ )
        {

            if (pic[idx++])
            {
                cout << "1";
            }
            else
            {
                cout << " ";
            }
        }

        cout << endl;
    }
}

inline void preProcessInputData(const unsigned char src[], double out[], int size)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = (src[i] >= 128) ? 1.0 : 0.0;
    }
}

inline void preProcessInputDataWithNoise(const unsigned char src[], double out[], int size)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = ((src[i] >= 128) ? 1.0 : 0.0) + RandFloat() * 0.1;
    }
}

inline void preProcessInputData(const unsigned char src[],int size, InputIndex& indexs)
{
    for (int i = 0; i < size; i++)
    {
        if (src[i] >= 128)
        {
            indexs.push_back(i);
        }
    }
}


double trainEpoch(dataInput& src, bpNeuronNet& bpnn, int imageSize, int numImages)
{
    double net_target[NUM_NET_OUT];
    char* temp = new char[imageSize];
    progressDisplay progd(numImages);

    double* net_train = new double[imageSize];
    for (int i = 0; i < numImages; i++)
    {
        int label = 0;
        memset(net_target, 0, NUM_NET_OUT * sizeof(double));

        if (src.read(&label, temp))
        {
            net_target[label] = 1.0;
            preProcessInputDataWithNoise((unsigned char*)temp, net_train, imageSize);
            bpnn.training(net_train, net_target);

        }
        else
        {
            cout << "read train data failed" << endl;
            break;
        }
        //progd.updateProgress(i);

        progd++;
    }

    cout << "the error is:" << bpnn.getError() << " after training " << endl;

    delete []net_train;
    delete []temp;

    return bpnn.getError();
}

int testRecognition(dataInput& testData, bpNeuronNet& bpnn, int imageSize, int numImages)
{
    int ok_cnt = 0;
    double* net_out = NULL;
    char* temp = new char[imageSize];
    progressDisplay progd(numImages);
    double* net_test = new double[imageSize];
    for (int i = 0; i < numImages; i++)
    {
        int label = 0;

        if (testData.read(&label, temp))
        {			
            preProcessInputData((unsigned char*)temp, net_test, imageSize);
            bpnn.process(net_test, &net_out);

            int idx = -1;
            double max_value = -99999;
            for (int i = 0; i < NUM_NET_OUT; i++)
            {
                if (net_out[i] > max_value)
                {
                    max_value = net_out[i];
                    idx = i;
                }
            }

            if (idx == label)
            {
                ok_cnt++;
            }

            progd.updateProgress(i);
        }
        else
        {
            cout << "read test data failed" << endl;
            break;
        }
    }


    delete []net_test;
    delete []temp;

    return ok_cnt;

}


double trainEpoch2(dataInput& src, bpNeuronNet& bpnn, int imageSize, int numImages)
{
    double net_target[NUM_NET_OUT];
    char* temp = new char[imageSize];
    progressDisplay progd(numImages);

    InputIndex indexs;

    for (int i = 0; i < numImages; i++)
    {
        int label = 0;
        memset(net_target, 0, NUM_NET_OUT * sizeof(double));
        indexs.clear();

        if (src.read(&label, temp))
        {
            net_target[label] = 1.0;
            preProcessInputData((unsigned char*)temp, imageSize, indexs);

            bpnn.training(indexs.data(), indexs.size(), net_target);

        }
        else
        {
            cout << "read train data failed" << endl;
            break;
        }
        //progd.updateProgress(i);

        progd++;
    }

    cout << "the error is:" << bpnn.getError() << " after training " << endl;

    delete[]temp;

    return bpnn.getError();
}

int testRecognition2(dataInput& testData, bpNeuronNet& bpnn, int imageSize, int numImages)
{
    int ok_cnt = 0;
    double* net_out = NULL;
    char* temp = new char[imageSize];
    progressDisplay progd(numImages);
    InputIndex indexs;

    for (int i = 0; i < numImages; i++)
    {
        int label = 0;
        indexs.clear();

        if (testData.read(&label, temp))
        {
            preProcessInputData((unsigned char*)temp, imageSize, indexs);
            bpnn.process(indexs.data(), indexs.size(), &net_out);

            int idx = -1;
            double max_value = -99999;
            for (int i = 0; i < NUM_NET_OUT; i++)
            {
                if (net_out[i] > max_value)
                {
                    max_value = net_out[i];
                    idx = i;
                }
            }

            if (idx == label)
            {
                ok_cnt++;
            }

            progd.updateProgress(i);
        }
        else
        {
            cout << "read test data failed" << endl;
            break;
        }
    }

    delete[]temp;

    return ok_cnt;

}




int main(int argc, char* argv[])
{
    dataInput src;
    dataInput testData;
    bpNeuronNet* bpnn = NULL;
    srand((int)time(0));

    if (src.openImageFile(TRAIN_IMAGES_URL) && src.openLabelFile(TRAIN_LABELS_URL))
    {
        int imageSize = src.imageLength();
        int numImages = src.numImage();
        int epochMax = 1;

        double expectErr = 0.1;
#if 0
        char* temp = new char[imageSize];
        for (size_t i = 0; i < 5; i++)
        {
            if (src.readImage(temp))
            {
                showNumber((unsigned char*)temp, src.imageWidth(), src.imageHeight());
            }
        }
#endif


        bpnn = new bpNeuronNet(imageSize, NET_LEARNING_RATE);

        /** add first hidden layer */

		bpnn->addNeuronLayer(30);
		//bpnn->addNeuronLayer(NUM_HIDDEN);
        
        /** add output layer */
        bpnn->addNeuronLayer(NUM_NET_OUT);

        cout << "start training ANN..." << endl;
        uint64_t st = timeNowMs();

        for (int i = 0; i < epochMax; i++)
        {
            double err = trainEpoch(src, *bpnn, imageSize, numImages);

            //if (err <= expectErr)
            {
            //	cout << "train success,the error is: " << err << endl;
            //	break;
            }

            src.reset();
        }

        cout << "training ANN success...cast time: " << (timeNowMs() - st) << "(millisecond)" << endl;

        showSeparatorLine('=', 80);
        st = timeNowMs();
        
        if (testData.openImageFile(TEST_IMANGES_URL) && testData.openLabelFile(TEST_LABELS_URL))
        {
            imageSize = testData.imageLength();
            numImages = testData.numImage();
            
            cout << "start test ANN with t10k images..." << endl;

            int ok_cnt = testRecognition(testData, *bpnn, imageSize, numImages);

            cout << "digital recognition cast time:"
                << (timeNowMs() - st) << "(millisecond), " 
                <<  "ok_cnt: " << ok_cnt << ", total: " << numImages << endl;
        }
        else
        {
            cout << "open test image file failed" << endl;
        }


    }
    else
    {
        cout << "open train image file failed" << endl;
    }

    if (bpnn)
    {
        delete bpnn;
    }

    getchar();

    return 0;
}