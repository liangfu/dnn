
#include "test_precomp.hpp"

#include <string>
#include <fstream>
#include <iostream>

#include "cvext_c.h"

// using namespace std;

class CV_NetworkTest : public cvtest::BaseTest
{
public:
  CV_NetworkTest();
  ~CV_NetworkTest();

protected:
  void run(int);

  int TestTrainPredict(int test_num);
  int TestSaveLoad();

  int checkPredictError(int test_num);
  int checkLoadSave();

  // CvNetwork * network;
};

CV_NetworkTest::CV_NetworkTest()
{
}

CV_NetworkTest::~CV_NetworkTest()
{
}

int CV_NetworkTest::TestTrainPredict(int test_num)
{
  int code = cvtest::TS::OK;
  // if (!tmp_code){
  //   ts->printf( cvtest::TS::LOG, "Model training was failed.");
  //   return cvtest::TS::FAIL_INVALID_OUTPUT;
  // }
  code = checkPredictError(test_num);
  return code;
}

int CV_NetworkTest::checkPredictError(int test_num)
{
  // if (!gtb){
  //   return cvtest::TS::FAIL_GENERIC;
  // }
  // if ( abs( current_error - mean[test_num]) > 6*sigma[test_num] ){
  //   ts->printf( cvtest::TS::LOG, "Test error is out of range:\n"
  //               "abs(%f/*curEr*/ - %f/*mean*/ > %f/*6*sigma*/",
  //               current_error, mean[test_num], 6*sigma[test_num] );
  //   return cvtest::TS::FAIL_BAD_ACCURACY;
  // }
  return cvtest::TS::OK;
}


int CV_NetworkTest::TestSaveLoad()
{
  int code = cvtest::TS::OK;
  // if (!gtb){
  //   return cvtest::TS::FAIL_GENERIC;
  // }
  // model_file_name1 = cv::tempfile();
  // model_file_name2 = cv::tempfile();
  // gtb->save(model_file_name1.c_str());
  // gtb->calc_error(data, CV_TEST_ERROR, &test_resps1);
  // gtb->load(model_file_name1.c_str());
  // gtb->calc_error(data, CV_TEST_ERROR, &test_resps2);
  // gtb->save(model_file_name2.c_str());

  return checkLoadSave();
}

int CV_NetworkTest::checkLoadSave()
{
  int code = cvtest::TS::OK;

  // 1. compare files
  // delete temporary files
  // remove( model_file_name1.c_str() );
  // remove( model_file_name2.c_str() );

  // 2. compare responses
  // CV_Assert( test_resps1.size() == test_resps2.size() );
  // vector<float>::const_iterator it1 = test_resps1.begin(), it2 = test_resps2.begin();
  // for( ; it1 != test_resps1.end(); ++it1, ++it2 ){
  //   if( fabs(*it1 - *it2) > FLT_EPSILON ){
  //     ts->printf( cvtest::TS::LOG,
  //                 "Responses predicted before saving and after loading are different" );
  //     code = cvtest::TS::FAIL_INVALID_OUTPUT;
  //   }
  // }
  return code;
}



void CV_NetworkTest::run(int)
{
  std::string dataPath = std::string(ts->get_data_path());
  int code = cvtest::TS::OK;

  // for (int i = 0; i < 4; i++){
  //   int temp_code = TestTrainPredict(i);
  //   if (temp_code != cvtest::TS::OK){
  //     code = temp_code;break;
  //   }else if (i==0){
  //     temp_code = TestSaveLoad();
  //     if (temp_code != cvtest::TS::OK){code = temp_code;}
  //     delete data;
  //     data = 0;
  //   }
  //   delete gtb;gtb = 0;
  // }
  // delete data;
  // data = 0;

  ts->set_failed_test_info( code );
}

/////////////////////////////////////////////////////////////////////////////
//////////////////// test registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// TEST(ML_Network, accuracy) { CV_NetworkTest test; test.safe_run(); }

typedef void (*CvActivationFunc)(CvMat *, CvMat *);
void cvActivationGradCheck(CvActivationFunc actfunc, CvActivationFunc actfunc_der)
{
  int nr=100, nc=100;
  const float eps = 1e-4f;
  CvMat * src = cvCreateMat(nr,nc,CV_32F);
  CvMat * src_more = cvCreateMat(nr,nc,CV_32F);
  CvMat * src_less = cvCreateMat(nr,nc,CV_32F);
  CvMat * dst = cvCreateMat(nr,nc,CV_32F); 
  CvMat * dst_more = cvCreateMat(nr,nc,CV_32F); 
  CvMat * dst_less = cvCreateMat(nr,nc,CV_32F); 
  CvMat * diff = cvCreateMat(nr,nc,CV_32F); 
  CvMat * grad = cvCreateMat(nr,nc,CV_32F);
  CvMat * src_der = cvCreateMat(nr,nc,CV_32F);
  CvMat * error = cvCreateMat(nr,nc,CV_32F);
  CvRNG rng = cvRNG(-1);
  cvRandArr(&rng,src,CV_RAND_UNI,cvScalar(-5),cvScalar(5));
  cvAddS(src,cvScalar(eps),src_more);
  cvAddS(src,cvScalar(-eps),src_less);
  cvZero(dst); cvZero(dst_more); cvZero(dst_less);
  actfunc(src,dst);
  actfunc(src_more,dst_more);
  actfunc(src_less,dst_less);
  cvSub(dst_more,dst_less,diff);
  cvScale(diff,grad,1./(2.f*eps));
  actfunc_der(src,src_der);
  cvSub(grad,src_der,error);
  EXPECT_LT(cvAvg(error).val[0], 1e-5);
}

TEST(ML_Tanh, gradcheck){cvActivationGradCheck(cvTanh, cvTanhDer);}
TEST(ML_Sigmoid, gradcheck){cvActivationGradCheck(cvSigmoid, cvSigmoidDer);}
TEST(ML_ReLU, gradcheck){cvActivationGradCheck(cvReLU, cvReLUDer);}
TEST(ML_Softmax, gradcheck){cvActivationGradCheck(cvSoftmax, cvSoftmaxDer);}












