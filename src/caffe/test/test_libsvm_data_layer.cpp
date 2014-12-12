#include <map>
#include <string>
#include <vector>
#include <numeric>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LIBSVMDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LIBSVMDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    MakeTempFilename(&filename_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create a Vector of files with labels
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 8; ++i) {
      int label = (i % 3) - 1;
      outfile << label << " ";
      for (int j = 0; j < i + 2; j++) {
        outfile << j*(i+1) << ":1.0 ";
      }
      outfile << endl;
    }
    outfile.close();
  }

  virtual ~LIBSVMDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LIBSVMDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(LIBSVMDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  LIBSVMDataParameter* libsvm_data_param = param.mutable_libsvm_data_param();
  libsvm_data_param->set_batch_size(5);
  libsvm_data_param->set_source(this->filename_.c_str());
  libsvm_data_param->set_shuffle(false);
  libsvm_data_param->set_channels(65);
  LIBSVMDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 65);
  EXPECT_EQ(this->blob_top_data_->height(), 1);
  EXPECT_EQ(this->blob_top_data_->width(), 1);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  // Go through the data twice
  Blob<Dtype>* td = this->blob_top_data_;
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int k = 0; k < 5; ++k) {
      int i = (k + 5 * iter) % 8;

      for (int j = 0; j < i + 2; ++j) {
        EXPECT_EQ(td->data_at(k, (i+1) * j, 0, 0), 1);
      }
      int nonzero = std::accumulate(
          td->cpu_data() + td->offset(k),
          td->cpu_data() + td->offset(k + 1), 0.0);
      EXPECT_EQ(nonzero, i + 2);
      EXPECT_EQ(this->blob_top_label_->cpu_data()[k], i % 3 - 1);
    }
  }
}

/*
TYPED_TEST(LIBSVMDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  LIBSVMDataParameter* libsvm_data_param = param.mutable_libsvm_data_param();
  libsvm_data_param->set_batch_size(5);
  libsvm_data_param->set_source(this->filename_.c_str());
  libsvm_data_param->set_new_height(256);
  libsvm_data_param->set_new_width(256);
  libsvm_data_param->set_shuffle(false);
  LIBSVMDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(LIBSVMDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  LIBSVMDataParameter* libsvm_data_param = param.mutable_libsvm_data_param();
  libsvm_data_param->set_batch_size(5);
  libsvm_data_param->set_source(this->filename_.c_str());
  libsvm_data_param->set_shuffle(true);
  LIBSVMDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}
*/

}  // namespace caffe
