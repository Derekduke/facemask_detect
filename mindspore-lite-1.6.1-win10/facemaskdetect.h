
float max_value(float a , float b)
{
  if(a >= b)
    return a;
  else
    return b;
}

float min_value(float a , float b)
{
  if(a >= b)
    return b;
  else
    return a;
}

struct yolo_box
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};
struct conf_id
{
  float conf;
  int id;
  struct yolo_box box;
};