#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <math.h>
#include <omp.h>

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
//int n_epochs = 1;
int n_epochs = 100000;
int n_hidden = 128;
//int n_samples = 1;
int n_samples = 3;
float ppralpha = 0.85f;

ull total_steps;
ull step = 0;

ull nv = 0; //节点的个数
ull nt = 0; //网络的个数
int *ne; //每个网络的边个数
int **offsets;
int **edges;
float **weights;
int **weight_js;
int **degrees;

float *w0;
float *w1;
float *l; //关系表征



const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);

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

void init_sigmoid_table() {
  float x;
  sigmoid_table = static_cast<float *>(
    aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++) {
    x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float FastSigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  else if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline int irand(int max) { return lrand() % max; }

void init_walker(int n, int *j, float *probs) { // 采用的是alias sampling。j中存的是表中第i列不是事件i的另一个事件标号，probs存储的是i列中i事件对应的概率，采样过程使用首先采样某一列，再采样某一个特定的元素
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

int walker_draw(const int n, float *q, int *j) {
  int kk = int(floor(drand() * n));
  return drand() < q[kk] ? kk : j[kk];
}

inline int sample_neighbor(int node, int cur_t) {
  if (offsets[cur_t][node] == offsets[cur_t][node + 1])
    return -1;
  return edges[cur_t][offsets[cur_t][node] + walker_draw(degrees[cur_t][node], &weights[cur_t][offsets[cur_t][node]], &weight_js[cur_t][offsets[cur_t][node]])];
}

inline int sample_rw(int node, int cur_t) {
  int n2 = node;
  while (drand() < ppralpha) {
    int neighbor = sample_neighbor(n2, cur_t);
    if (neighbor == -1)
      return n2;
    n2 = neighbor;
  }
  return n2;
}

inline void update(float *w_s, float *w_t, float *l, int label, const float bias) {
  float score = -bias;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c] * l[c];
  score = (label - FastSigmoid(score)) * global_lr;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t[c] += score * w_s[c] * l[c];
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c] * l[c];
AVX_LOOP
  for (int c = 0; c < n_hidden; c++) {
    l[c] += score * w_s[c] * w_t[c];
    if (l[c] > 1) {
        l[c] = 1;
    }
    else {
        if(l[c] < 0) {
            l[c] = 0;
        }
    }
  }
}

void Train() {
#pragma omp parallel num_threads(n_threads)
  {
    const float nce_bias = log(nv);
    const float nce_bias_neg = log(nv / float(n_samples));
    int tid = omp_get_thread_num();
    ull last_ncount = 0;
    ull ncount = 0;
    float lr = global_lr;
#pragma omp barrier
    while (1) {
      if (ncount - last_ncount > 10000) {
        ull diff = ncount - last_ncount;
#pragma omp atomic
        step += diff;
        if (step > total_steps)
          break;
        if (tid == 0)
          if (!silent)
            cout << fixed << "\r Progress " << std::setprecision(2)
                 << step / (float)(total_steps + 1) * 100 << "%";
        last_ncount = ncount;
      }
      size_t n1 = irand(nv);
      int cur_t = irand(nt);
      size_t n2 = sample_rw(n1, cur_t);
      update(&w0[n1 * n_hidden], &w1[n2 * n_hidden], &l[cur_t * n_hidden], 1, nce_bias);
      for (int i = 0; i < n_samples; i++) {
        size_t neg = irand(nv);
        update(&w0[n1 * n_hidden], &w1[neg * n_hidden], &l[cur_t * n_hidden], 0, nce_bias_neg);
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
  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    cout << "Input file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else {
    cout << "Output file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-silent"), argc, argv)) > 0)
    silent = true;
  if ((a = ArgPos(const_cast<char *>("-nsamples"), argc, argv)) > 0)
    n_samples = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-steps"), argc, argv)) > 0)
    n_epochs = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    global_lr = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-alpha"), argc, argv)) > 0)
    ppralpha = atof(argv[a + 1]);

  ifstream embFile(network_file, ios::in | ios::binary);
  if (embFile.is_open()) {
    char header[] = "----";
    embFile.seekg(0, ios::beg);
    embFile.read(header, 4);
    if (strcmp(header, "XGFS") != 0) {
      cout << "Invalid header!: " << header << endl;
      return 1;
    }
    embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long)); //节点个数
    embFile.read(reinterpret_cast<char *>(&nt), sizeof(long long)); //网络个数
    cout<<"node = "<<nv<<". type = "<< nt<< endl;
    ne = static_cast<int *>(aligned_malloc(nt * sizeof(int32_t), DEFAULT_ALIGN)); // 存储每个to_node,这个和offset配合使用
    embFile.read(reinterpret_cast<char *>(ne), nt * sizeof(int32_t));
    for(int i=0; i<nt; i++){
      cout<<i<<":"<<ne[i]<<endl;
    }
    offsets = new int*[nt];
    for(int i=0; i<nt; i++){
      offsets[i] = static_cast<int *>(aligned_malloc((nv+1) * sizeof(int32_t), DEFAULT_ALIGN));
      embFile.read(reinterpret_cast<char *>(offsets[i]), nv*sizeof(int32_t));
      offsets[i][nv] = int(ne[i]);
    }
    edges = new int*[nt];
    for(int i=0; i<nt; i++){
      edges[i] = static_cast<int *>(aligned_malloc(ne[i] * sizeof(int32_t), DEFAULT_ALIGN));
      embFile.read(reinterpret_cast<char *>(edges[i]), ne[i] * sizeof(int32_t));
    }
    weights = new float*[nt];
    for(int i=0; i<nt; i++){
      weights[i] = static_cast<float *>(aligned_malloc(ne[i] * sizeof(float), DEFAULT_ALIGN));
      embFile.read(reinterpret_cast<char *>(weights[i]), ne[i]*sizeof(float));
    }
    embFile.close();
  } else {
    return 0;
  }
  weight_js = new int*[nt];
  for(int i=0; i<nt; i++){
    weight_js[i] = static_cast<int *>(aligned_malloc(ne[i] * sizeof(int32_t), DEFAULT_ALIGN));
    memset(weight_js[i], 0, ne[i]*sizeof(int32_t));
  }
  //初始化学习向量,用1/n对l进行初始化
  w0 = static_cast<float *>(aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  w1 = static_cast<float *>(aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  for (size_t i = 0; i < nv * n_hidden; i++)
    w0[i] = drand() - 0.5;
  for (size_t i = 0; i < nv * n_hidden; i++)
    w1[i] = drand() - 0.5;
  l = static_cast<float *>(aligned_malloc(nt * n_hidden * sizeof(float), DEFAULT_ALIGN));
  for (size_t i = 0; i < nt * n_hidden; i++)
    l[i] = 1.0/n_hidden;
  degrees = new int*[nt];
  for(int i=0; i<nt; i++){
    degrees[i] = (int *)malloc(nv * sizeof(int)); //存储每个节点所有的出度
    for(int j=0; j<nv; j++){
      degrees[i][j] = offsets[i][j+1] - offsets[i][j];
    }
  }

  for(int t=0; t<nt; t++){
    for(int i=0; i<nv; i++){
      float sum = 0;
      for(int j=offsets[t][i]; j<offsets[t][i+1]; j++){
        sum += weights[t][j];
      }
      sum /= degrees[t][i];
      for(int j=offsets[t][i]; j<offsets[t][i+1]; j++){
        weights[t][j] /= sum;
      }
      init_walker(degrees[t][i], &weight_js[t][offsets[t][i]], &weights[t][offsets[t][i]]);
    }
  }

  total_steps = nt * n_epochs * (long long)nv;
  cout << "Total steps (mil): " << total_steps / 1000000. << endl;
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  Train();
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  cout << endl
       << "Calculations took "
       << chrono::duration_cast<std::chrono::duration<float>>(end - begin)
              .count()
       << " s to run" << endl;
  for (size_t i = 0; i < nv * n_hidden; i++)
    if (w0[0] != w0[0]) {
      cout << endl << "NaN! Not saving the result.." << endl;
      return 1;
    }
  ofstream output(embedding_file+"_s", std::ios::binary);
  output.write(reinterpret_cast<char *>(w0), sizeof(float) * n_hidden * nv);
  output.close();

  ofstream output2(embedding_file+"_t", std::ios::binary);
  output2.write(reinterpret_cast<char *>(w1), sizeof(float) * n_hidden * nv);
  output2.close();

  ofstream output3(embedding_file+"_l", std::ios::binary);
  output3.write(reinterpret_cast<char *>(l), sizeof(float) * n_hidden * nt);
  output3.close();

 }

