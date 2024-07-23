#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdint.h>

using namespace std;

#if defined(__AVX2__) || \
defined(__FMA__)
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0
#define DEFAULT_ALIGN 128

typedef unsigned long long ull;

bool silent = false;
int n_threads = 1;
float global_lr = 0.0025f;
int n_epochs = 100000;
int n_hidden = 128;
int n_samples = 3;
float ppralpha = 0.85f;

ull total_steps;
ull step = 0;

ull nv = 0, ne = 0;
int *offsets;
int *edges;
float *weights;
int *weight_js;
int *degrees;

float *w0;

const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f); //1024/6*2

uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// http://xoroshiro.di.unimi.it/#shootout
uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  rng_seed[1] = rotl(s1, 36);                   // c
  return result;
}

static inline double drand() {  
  '''
  rand 返回介于 0.0 到 1.0 之间的实数值。
  drand 返回介于 0.0 到 1.0 之间的双精度值。
  irand 返回介于 0 到 2147483647 之间的正整数。
  '''
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline void *aligned_malloc(
  size_t size,
  size_t align) {
#ifndef _MSC_VER
void *result;
if (posix_memalign(&result, align, size)) result = 0;
#else
void *result = _aligned_malloc(size, align);
#endif
return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
_aligned_free(ptr);
#else
free(ptr);
#endif
}

void init_sigmoid_table() { // 初始化sigmoid表,sigmoid表格中存储了一些离散的sigmoid值
  float x;
  sigmoid_table = static_cast<float *>(
    aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++) {
    x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float FastSigmoid(float x) { // 快速sgimoid计算,直接在sigmoid表格中查找
  if (x > SIGMOID_BOUND)
    return 1;
  else if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; } // min到max之间的一个数,生成

inline int irand(int max) { return lrand() % max; } // 生成一个不大于max的数

// init_walker(degrees[i], &weight_js[offsets[i]], &weights[offsets[i]])
void init_walker(int n, int *j, float *probs) { // assumes probs are normalized
  vector<int> smaller, larger;
  for (int i = 0; i < n; i++) {
    if (probs[i] < 1)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }
  while (smaller.size() != 0 && larger.size() != 0) {
    int small = smaller.back();
    smaller.pop_back();
    int large = larger.back();
    larger.pop_back();
    j[small] = large;
    probs[large] += probs[small] - 1;
    if (probs[large] < 1)
      smaller.push_back(large);
    else
      larger.push_back(large);
  }
}

int walker_draw(const int n, float *q, int *j) { // 作用是什么?
  int kk = int(floor(drand() * n));
  return drand() < q[kk] ? kk : j[kk];
}

inline int sample_neighbor(int node) { // 从node的邻居中随机取一个
  if (offsets[node] == offsets[node + 1]) // node 节点没有邻居
    return -1;
  return edges[offsets[node] + walker_draw(degrees[node], &weights[offsets[node]], &weight_js[offsets[node]])];
}

inline int sample_rw(int node) { // 随机游走??
  int n2 = node;
  while (drand() < ppralpha) { //随机产生的一个0到1之间的值与ppralpha进行比较

    int neighbor = sample_neighbor(n2); //从n2节点随机一个邻居
    if (neighbor == -1) //随机选择邻居有一定的概率是返回-1的,如果是返回-1,那么直接返回
      return n2;
    n2 = neighbor; //否则以n2作为当前节点继续随机
  }
  return n2; //最后返回n2节点
}

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

inline void update(float *w_s, float *w_t, int label, const float bias) {
  // 对应于算法中的 UpdateBypair(u,v,k,D)
  // *w_s: n1的embedding?
  // *w_t: n2的embedding?
  // label: 1
  // bias: log(nv),节点个数的对数
  float score = -bias; // log(c)* \phi_n呢?
AVX_LOOP
  score += w_s[c] * w_t[c]; // 两个向量的内积
  score = (label - FastSigmoid(score)) * global_lr; 
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t[c] += score * w_s[c];
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c];
}

void Train() {
#pragma omp parallel num_threads(n_threads)
  {
    const float nce_bias = log(nv); 
    const float nce_bias_neg = log(nv / float(n_samples));
    int tid = omp_get_thread_num();
    // ncount 和last_ncount 分别是什么 ?
    ull last_ncount = 0;  // unsigned long long  ull
    ull ncount = 0; 
    float lr = global_lr; //学习率
#pragma omp barrier // barrier：用于并行域内代码的线程同步，线程执行到barrier时要停下等待，直到所有线程都执行到barrier时才继续往下执行；
    while (1) {
      if (ncount - last_ncount > 10000) { //进行了10000步之后才会执行这一部分,主要用于在命令行输出
        ull diff = ncount - last_ncount;
#pragma omp atomic // atomic：用于指定一个数据操作需要原子性地完成；
        step += diff; //该步骤需要原子性地完成
        if (step > total_steps) 
          break;
        if (tid == 0)
          if (!silent) 
            cout << fixed << "\r Progress " << std::setprecision(2)
                 << step / (float)(total_steps + 1) * 100 << "%"; // progress 95.4% 
        last_ncount = ncount;
      }
      size_t n1 = irand(nv);    // 随机产生一个节点n1作为游走开始的节点
      size_t n2 = sample_rw(n1); // 随机游走,随机产生n1的一个若干跳的邻居节点
      '''
      w0中存储的是一些embedding
      &w0[n1 * n_hidden]:n1节点的embedding?
      &w0[n2 * n_hidden]:n2 节点的embedding?
      最终的embedding是长度是64,为什么这里是128呢? 答: 在主函数中重新赋值了
      '''
      update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, nce_bias);
      for (int i = 0; i < n_samples; i++) { //负采样
        size_t neg = irand(nv); // 随机负采样一个节点
        update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, nce_bias_neg); //由于是负样本,因此label应该为0
      }
      ncount++;
    }
  }
}

int main(int argc, char **argv) {
  int a = 0;
  string network_file, embedding_file;
  ull x = time(nullptr); // 每次根据时间生成两个不同的随机种子
  for (int i = 0; i < 2; i++) {
    ull z = x += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ z >> 31;
  }
  init_sigmoid_table();
  /* ######处理输入的数据############*/
  // 输入的图
  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0) 
    network_file = argv[a + 1]; 
  else {
    cout << "Input file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  // 输出位置
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else {
    cout << "Output file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  // embedding的维度.默认为128
  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  // 未知?
  if ((a = ArgPos(const_cast<char *>("-silent"), argc, argv)) > 0)
    silent = true;
  // 负采样个数,默认为3
  if ((a = ArgPos(const_cast<char *>("-nsamples"), argc, argv)) > 0)
    n_samples = atoi(argv[a + 1]);
  // 训练时线程个数,默认是1
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  // epoch个数,默认为100000
  if ((a = ArgPos(const_cast<char *>("-steps"), argc, argv)) > 0)
    n_epochs = atoi(argv[a + 1]);
  // 学习率
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    global_lr = atof(argv[a + 1]);
  // ppralpha,跳转的概率
  if ((a = ArgPos(const_cast<char *>("-alpha"), argc, argv)) > 0)
    ppralpha = atof(argv[a + 1]);
  // 打开图文件,edgelist格式
  ifstream embFile(network_file, ios::in | ios::binary);
  if (embFile.is_open()) {
    char header[] = "----";
    embFile.seekg(0, ios::beg);
    embFile.read(header, 4);
    // 检查文件的有效性?
    if (strcmp(header, "XGFS") != 0) {
      cout << "Invalid header!: " << header << endl;
      return 1;
    }
    // 对nv和ne 赋值
    embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
    embFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
    
    // 存储当前节点之前所有边数的个数，其中最后一个元素是offsets[-1] = ne 
    offsets = static_cast<int *>(aligned_malloc((nv + 1) * sizeof(int32_t), DEFAULT_ALIGN)); 
    // 存储所有的edge,这个和offset配合使用:根据offset的值,可以得到每个节点对应的边
    edges = static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN)); 
    // 存储跟edge对应的每条边的权重
    weights = static_cast<float *>(aligned_malloc(ne * sizeof(float), DEFAULT_ALIGN)); 
    // 对offsets进行赋值
    embFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(int32_t));
    offsets[nv] = (int)ne;
    // 对edge赋值
    embFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
    // 对weight赋值
    embFile.read(reinterpret_cast<char *>(weights), sizeof(float) * ne);
    cout << "nv: " << nv << ", ne: " << ne << endl;
    embFile.close();
  } else {
    return 0;
  }

  // weighted_js 是什么作用
  weight_js = static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN));
  memset(weight_js, 0, ne*sizeof(int32_t));
  // 初始化w0
  w0 = static_cast<float *>(aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  // 输出w0
  for (size_t i = 0; i < nv * n_hidden; i++){
    if(i+1 % n_hidden!=0){
      cout<<w0[i]<<" ";
    }
    else{
      cout<<w0[i]<<endl;
    }
    // 随机初始化w0
    w0[i] = drand() - 0.5;
  }
  // 随机初始化w0
  for (size_t i = 0; i < nv * n_hidden; i++){
    w0[i] = drand() - 0.5;
  }
  //存储每个节点所有的出度
  degrees = (int *)malloc(nv * sizeof(int)); 
  for (int i = 0; i < nv; i++)
    degrees[i] = offsets[i + 1] - offsets[i];
  // 处理weight
  for(int i=0;i<nv;i++)
  {
    float sum = 0;
    for(int j=offsets[i];j<offsets[i+1];j++)
      sum += weights[j];
    sum /= degrees[i]; // 平均的sum
    //将weight做归一化处理
    for(int j=offsets[i];j<offsets[i+1];j++) 
      weights[j] /= sum;
    
    // init_walker ??
    init_walker(degrees[i], &weight_js[offsets[i]], &weights[offsets[i]]);
  }

  // total_step: 随机游走的步数?
  total_steps = n_epochs * (long long)nv; 
  cout << "Total steps (mil): " << total_steps / 1000000. << endl;
  // 训练
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  Train();
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  cout << endl
       << "Calculations took "
       << chrono::duration_cast<std::chrono::duration<float>>(end - begin)
              .count()
       << " s to run" << endl;
  // 检查w0
  for (size_t i = 0; i < nv * n_hidden; i++)
    if (w0[0] != w0[0]) {
      cout << endl << "NaN! Not saving the result.." << endl;
      return 1;
    }
  int j=0;
  // 输出w0
  for (size_t i = 0; i < nv * n_hidden; i++){
    if(i+1 % n_hidden!=0){
      cout<<w0[i]<<" ";
    }
    else{
      cout<<w0[i]<<endl;
    }
    j = i;
    // w0[i] = drand() - 0.5;
  }
  // cout << j/n_hidden<<"dasdasd";
  // 二进制保存
  ofstream output(embedding_file, std::ios::binary);
  output.write(reinterpret_cast<char *>(w0), sizeof(float) * n_hidden * nv);
  output.close();
}
