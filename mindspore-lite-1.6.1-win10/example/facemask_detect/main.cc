#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "img_test.h"
#include "achors_data.h"
#include "facemaskdetect.h"

using mindspore::MSTensor;

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    std::cerr << "file is nullptr." << std::endl;
    return nullptr;
  }

  std::ifstream ifs(file, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    std::cerr << "file: " << file << " is not exist." << std::endl;
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cerr << "file: " << file << " open failed." << std::endl;
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cerr << "malloc buf failed, file: " << file << std::endl;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

/*可以在这里定义自己需要测试的图片和预处理方式*/
float input_data[260*260*3] = {0.0};
void GetImage(int size , void* model_input , const uint8_t* input_image)
{
  for(uint32_t i=0 ; i<260*260*3 ; i++)
  {
    input_data[i] = (float)input_image[i]/255.0;
  }
  memcpy(model_input , input_data ,  size);
}
/*相关全局变量定义*/
float loc_data[23888]={0.0}; //output tensor 0 人脸的坐标框信息输出
float cls_data[11944]={0.0}; //output tensor 1 带口罩和不带口罩的分类信息输出
float bbox_max_scores[5972]={0.0}; //记录每个候选坐标框的置信度
int bbox_max_score_classes[5972]={0}; //记录每个候选坐标框最可能的分类结果
int conf_keep_idx_list[5972] = {0}; //最终选取的检测坐标框绝对id号
int pick_id=0; //最终选取的检测坐标框个数
struct yolo_box box_value[5972]={0.0,0.0,0.0,0.0}; //记录每个候选坐标框的四点坐标
float variances[4] = {0.1 , 0.1 , 0.2 , 0.2}; 
float predict_bbox[5972*4] = {0.0}; //yolo_decode输出的候选坐标框坐标记录
struct conf_id conf_id_sort[5972]; //将坐标框的置信度和绝对id号绑定在一起，防止排序后id号打乱
float area[5972]={0.0}; //记录坐标框的区域面积
float conf_thresh=0.2; //置信度阈值
float iou_thresh=0.5; //IOU阈值
char id2class[2][10]={"Mask","NoMask"}; //记录分类结果的标签
float* decode_bbox(float* anchors, float* raw_outputs, float* variances);//处理模型输出的坐标框
int* single_class_non_max_suppression(float* bboxes, float* confidences);//非极大值抑制剔除无用的坐标框

int main(int argc, const char **argv) {
  // Read model file.
  std::string model_path = "../model/face_mask_detection.ms";
  size_t size = 0;
  char *model_buf = ReadFile(model_path.c_str(), &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    delete[](model_buf);
    std::cerr << "New context failed." << std::endl;
    return -1;
  }
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  if (device_info == nullptr) {
    delete[](model_buf);
    std::cerr << "New CPUDeviceInfo failed." << std::endl;
    return -1;
  }
  device_list.push_back(device_info);

  // Create model
  auto model = new (std::nothrow) mindspore::Model();
  if (model == nullptr) {
    delete[](model_buf);
    std::cerr << "New Model failed." << std::endl;
    return -1;
  }

  // Build model
  auto build_ret = model->Build(model_buf, size, mindspore::kMindIR, context);
  delete[](model_buf);
  if (build_ret != mindspore::kSuccess) {
    delete model;
    std::cerr << "Build model error " << std::endl;
    return -1;
  }

  auto inputs = model->GetInputs();
  for (auto tensor : inputs) {
    auto input_data = reinterpret_cast<float*>(tensor.MutableData());
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      delete model;
      return -1;
    }
    GetImage(tensor.DataSize() , input_data , gImage_img_test);
  }

  // Predict
  //std::vector<MSTensor> outputs;
  auto outputs = model->GetOutputs();
  auto status = model->Predict(inputs, &outputs);
  if (status != mindspore::kSuccess) {
    std::cerr << "Inference error." << std::endl;
    delete model;
    return -1;
  }

  // Get Output Tensor Data.
  std::cout << "\n------- print outputs ----------" << std::endl;
  uint8_t cnt = 0;
  for (auto tensor : outputs) {
    std::cout << "out tensor name is:" << tensor.Name() << "\nout tensor size is:" << tensor.DataSize()
              << "\nout tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.MutableData());
    std::cout << "output data is:"<< std::endl;
    for (int i = 0; i < tensor.ElementNum(); i++) {
      if(cnt == 0)
      {
        loc_data[i] = out_data[i];
      }
      else
      {
        cls_data[i] = out_data[i];
      }
    }
    std::cout << std::endl;
    cnt++;
  }
  std::cout << "------- print end ----------\n" << std::endl;

  //进行数据后处理
  decode_bbox(anchors, loc_data, variances);
  for(int i=0 ; i<5972 ; i=i+1)
  {
    if(cls_data[2*i]>=cls_data[2*i+1])
    {
      bbox_max_scores[i] = cls_data[2*i];
      bbox_max_score_classes[i] = 0;
    }
    else
    {
      bbox_max_scores[i] = cls_data[2*i+1];
      bbox_max_score_classes[i] = 1;
    }
  }
  int* res = single_class_non_max_suppression(predict_bbox, bbox_max_scores);
  for(int idx=0 ; idx<pick_id ; idx++)
  {
      float conf = float(bbox_max_scores[res[idx]]);
      int class_id = bbox_max_score_classes[res[idx]];
      int xmin = (int)max_value(0 , predict_bbox[res[idx]*4+0]*260);
      int ymin = (int)max_value(0 , predict_bbox[res[idx]*4+1]*260);
      int xmax = (int)min_value(predict_bbox[res[idx]*4+2]*260, 260);
      int ymax = (int)min_value(predict_bbox[res[idx]*4+3]*260, 260);
      printf("xmin=%d , ymin=%d , xmax=%d , ymax=%d\n" , xmin , ymin , xmax , ymax);
      printf("conf=%f , class_id=%d , result=%s\n" , conf , class_id , id2class[class_id]);
  }
  // Delete model.
  delete model;
  return mindspore::kSuccess;
}

float* decode_bbox(float* anchors, float* raw_outputs, float* variances)
{
  for(int i=0 ; i<5972; i++)
  {
    float anchor_centers_x = (anchors[i*4+0]+anchors[i*4+2])/2;
    float anchor_centers_y = (anchors[i*4+1]+anchors[i*4+3])/2;
    float anchors_w = anchors[i*4+2]-anchors[i*4+0];
    float anchors_h = anchors[i*4+3]-anchors[i*4+1];
    float temp[4]={0.0};
    for(int j=0 ; j<4 ; j++)
    {
      temp[j] = raw_outputs[i*4+j]*variances[j];
    }
    float predict_center_x = temp[0]*anchors_w+anchor_centers_x;
    float predict_center_y = temp[1]*anchors_h+anchor_centers_y;
    float predict_w = exp(temp[2])*anchors_w;
    float predict_h = exp(temp[3])*anchors_h;
    float predict_xmin = predict_center_x - predict_w / 2;
    float predict_ymin = predict_center_y - predict_h / 2;
    float predict_xmax = predict_center_x + predict_w / 2;
    float predict_ymax = predict_center_y + predict_h / 2; 
    predict_bbox[i*4+0] = predict_xmin;
    predict_bbox[i*4+1] = predict_ymin;
    predict_bbox[i*4+2] = predict_xmax;
    predict_bbox[i*4+3] = predict_ymax;
  }
  return predict_bbox;
} 

//升序排列
static int nms_compare(const void* pa, const void* pb){
    conf_id a = *(conf_id *)pa;
    conf_id b = *(conf_id *)pb;
    float diff = a.conf-b.conf;
    if (diff < 0)
        return -1;
    else if (diff > 0)
        return 1;
    return 0;  
}

int* single_class_non_max_suppression(float* bboxes, float* confidences)
{
  int keep_top_k=-1;

  int cnt = 0;
  for(int i=0 ; i<5972 ; i++)
  {
    int conf_keep_idx = -1;
    if(confidences[i] > conf_thresh)
    {
      conf_keep_idx = i;
      struct conf_id conf_id_sort_temp;
      conf_id_sort_temp.conf = confidences[conf_keep_idx];
      conf_id_sort_temp.id = conf_keep_idx;
      conf_id_sort[cnt] = conf_id_sort_temp;
      struct yolo_box temp_box;
      temp_box.xmin = bboxes[i*4+0];
      temp_box.ymin = bboxes[i*4+1];
      temp_box.xmax = bboxes[i*4+2];
      temp_box.ymax = bboxes[i*4+3];
      box_value[cnt] = temp_box;
      //printf("xmin=%f, ymin=%f, xmax=%f, ymax=%f\n",box_value[cnt].xmin, box_value[cnt].ymin,box_value[cnt].xmax,box_value[cnt].ymax); 
      float area_temp = (temp_box.xmax - temp_box.xmin + 1e-3) * (temp_box.ymax - temp_box.ymin + 1e-3);
      area[cnt] = area_temp;
      cnt++;
    }
  }
  qsort(conf_id_sort, cnt, sizeof(conf_id), nms_compare);
  printf("confidences sort:\n");
  for(int i=0 ; i<cnt ; i++)
  {
      printf("%f " , (conf_id_sort[i]).conf);
  }
  printf("\n");
  int pick[5972]={0};
  pick_id=0;
  int cnt_remain = cnt;
  int del[cnt]={0};
  while(cnt_remain>0)
  {
    int last = cnt_remain-1;
    if(del[last] == 1)
    {
      cnt_remain--;
      continue;
    }
    int k=last;
    pick[pick_id] = k;
    pick_id++;
    if(keep_top_k != -1)
    {
      if(pick_id >= keep_top_k) break;
    }
    float overlap_xmin[cnt_remain-1]={0.0};
    for(int r=0 ; r<cnt_remain-1 ; r++)
    {
      float overlap_xmin_temp = max_value(box_value[k].xmin , box_value[r].xmin);
      float overlap_ymin_temp = max_value(box_value[k].ymin , box_value[r].ymin);
      float overlap_xmax_temp = min_value(box_value[k].xmax , box_value[r].xmax);
      float overlap_ymax_temp = min_value(box_value[k].ymax , box_value[r].ymax);
      float overlap_w_temp = max_value(0 , overlap_xmax_temp-overlap_xmin_temp);
      float overlap_h_temp = max_value(0 , overlap_ymax_temp-overlap_ymin_temp);
      float overlap_area_temp = overlap_w_temp * overlap_h_temp;
      float overlap_ratio = overlap_area_temp / (area[conf_id_sort[r].id] + area[k] - overlap_area_temp);
      if(overlap_ratio > iou_thresh)
      {
        del[r]= 1;
      }
    }
    cnt_remain--;
  }
  for(int m=0 ; m<pick_id ; m++)
  {
     conf_keep_idx_list[m] = (conf_id_sort[pick[m]]).id;
  }
  return conf_keep_idx_list;
}