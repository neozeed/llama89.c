/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include <stdarg.h>

#if defined(_WIN32) && defined(__BORLANDC__)
  #define NO_SIMD
  #define NO_FAST_MATH
  #define LEGACY_FP
  typedef DWORDLONG uint64;
#else
  typedef unsigned long long uint64;
#endif

#define REAL_EPSILON 1e-5f

#ifdef LEGACY_FP
static float safe_exp(float x) {
  // Avoid overflow
  if (x > 88.0) return 1e38;
  if (x < -88.0) return 0;
  return exp(x);
}

static float safe_sqrt(float x) {
  if (x <= 0) return 0;
  return sqrt(x);
}
#else
#define safe_exp expf
#define safe_sqrt sqrtf
#endif

static void log_debug(const char* format, ...) {
    static FILE* debug_file = NULL;
    va_list args;

    // Open file on first use
    if (!debug_file) {
        debug_file = fopen("debug.log", "w");
        if (!debug_file) return;
    }

    va_start(args, format);
    vfprintf(debug_file, format, args);
    va_end(args);

    // Flush after each write to ensure we get logs even if program crashes
    fflush(debug_file);
}

// Add at the top with other helper functions
static void read_weights_from_file(FILE* file, float* dest, size_t n_elements) {
  float* temp;
  size_t i, chunk_size, current_chunk, remaining;
  const size_t MAX_CHUNK = 16384; // Read in 16KB chunks

  if (sizeof(float) == sizeof(float)) {
    // Direct read for 32-bit platforms
    remaining = n_elements;
    while (remaining > 0) {
      chunk_size = (remaining < MAX_CHUNK) ? remaining : MAX_CHUNK;
      if (fread(dest, sizeof(float), chunk_size, file) != chunk_size) {
        log_debug("Failed to read chunk of %ld elements\n", (long)chunk_size);
        exit(EXIT_FAILURE);
      }
      dest += chunk_size;
      remaining -= chunk_size;
    }
  } else {
    // Read into temp buffer and convert for other platforms
    temp = (float*)malloc(MAX_CHUNK * sizeof(float));
    if (!temp) {
      log_debug("Failed to allocate temp buffer for %ld elements\n", (long)MAX_CHUNK);
      exit(EXIT_FAILURE);
    }

    remaining = n_elements;
    while (remaining > 0) {
      current_chunk = (remaining < MAX_CHUNK) ? remaining : MAX_CHUNK;

      if (fread(temp, sizeof(float), current_chunk, file) != current_chunk) {
        log_debug("Failed to read chunk of %ld elements (remaining: %ld)\n",
                 (long)current_chunk, (long)remaining);
        free(temp);
        exit(EXIT_FAILURE);
      }

      for (i = 0; i < current_chunk; i++) {
        dest[i] = (float)temp[i];
      }

      dest += current_chunk;
      remaining -= current_chunk;
    }

    free(temp);
  }
}

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    long file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    printf("Allocating RunState buffers: dim=%d, hidden_dim=%d, kv_dim=%d\n",
           p->dim, p->hidden_dim, kv_dim);

    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("All RunState buffers allocated successfully\n");
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, long* file_size) {
    FILE *file;
    int shared_weights;
    size_t offset;
    int head_size;
    size_t layer_size;
    size_t embedding_size;
    const size_t chunk_size = 64 * 1024; // 64KB chunks
    size_t remaining, current_chunk;
    char* ptr;
    long current_pos;
    size_t bytes_read;
    size_t read_size;
    size_t rms_size;
    size_t total_allocated, alloc_size;
    const size_t MAX_SINGLE_ALLOC = 1024 * 1024; // 1MB max single allocation

    file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    head_size = config->dim / config->n_heads;

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    log_debug("Reading checkpoint: file_size=%ld bytes (%.2f MB)\n",
              *file_size, (float)*file_size / (1024*1024));
    log_debug("Config: vocab_size=%d, dim=%d, n_layers=%d\n",
              config->vocab_size, config->dim, config->n_layers);

    // Skip the config header for subsequent reads
    offset = sizeof(Config);

    // token embedding table
    embedding_size = config->vocab_size * config->dim;
    log_debug("Allocating token_embedding_table: %ld elements (%ld bytes)\n",
              (long)embedding_size, (long)(embedding_size * sizeof(float)));

    weights->token_embedding_table = (float*)malloc(embedding_size * sizeof(float));
    if (!weights->token_embedding_table) {
        log_debug("Failed to allocate token_embedding_table\n");
        exit(EXIT_FAILURE);
    }
    log_debug("Allocated token_embedding_table at %p\n", (void*)weights->token_embedding_table);

    // Read in chunks
    remaining = embedding_size * sizeof(float);
    ptr = (char*)weights->token_embedding_table;
    fseek(file, offset, SEEK_SET);

    while (remaining > 0) {
        current_pos = ftell(file);
        current_chunk = remaining < chunk_size ? remaining : chunk_size;

        bytes_read = fread(ptr, 1, current_chunk, file);
        if (bytes_read != current_chunk) {
            log_debug("Failed to read chunk: expected %ld bytes, got %ld bytes\n",
                     (long)current_chunk, (long)bytes_read);
            log_debug("File position: %ld, file size: %ld\n",
                     ftell(file), *file_size);
            exit(EXIT_FAILURE);
        }

        ptr += current_chunk;
        remaining -= current_chunk;
    }

    offset += embedding_size * sizeof(float);
    log_debug("Successfully read token_embedding_table in chunks\n");

    // rms attention weights
    rms_size = config->n_layers * config->dim;
    weights->rms_att_weight = (float*)malloc(rms_size * sizeof(float));
    if (!weights->rms_att_weight) {
        log_debug("Failed to allocate rms_att_weight\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->rms_att_weight, rms_size);
    offset += rms_size * sizeof(float);

    // wq, wk, wv weights
    layer_size = config->dim * config->n_heads * head_size;
    weights->wq = (float*)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug("Allocated wq at %p\n", (void*)weights->wq);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wq, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    layer_size = config->dim * config->n_kv_heads * head_size;
    weights->wk = (float*)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug("Allocated wk at %p\n", (void*)weights->wk);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wk, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    weights->wv = (float*)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug("Allocated wv at %p\n", (void*)weights->wv);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wv, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    // wo weights
    layer_size = config->n_heads * head_size * config->dim;
    weights->wo = (float*)malloc(config->n_layers * layer_size * sizeof(float));
    log_debug("Allocated wo at %p\n", (void*)weights->wo);
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->wo, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    // Remaining weights...
    weights->rms_ffn_weight = (float*)malloc(config->n_layers * config->dim * sizeof(float));
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->rms_ffn_weight, config->n_layers * config->dim);
    offset += config->n_layers * config->dim * sizeof(float);

    // w1, w2, w3 weights
    layer_size = config->dim * config->hidden_dim;
    log_debug("Allocating w1: layer_size=%ld, total bytes=%ld\n",
              (long)layer_size, (long)(config->n_layers * layer_size * sizeof(float)));

    // Add memory tracking
    total_allocated = 0;

    // When allocating large buffers:
    alloc_size = config->n_layers * layer_size * sizeof(float);
    if (alloc_size > MAX_SINGLE_ALLOC) {
      log_debug("Warning: Large allocation of %ld bytes requested\n", (long)alloc_size);
    }

    weights->w1 = (float*)malloc(alloc_size);
    if (!weights->w1) {
        log_debug("Failed to allocate w1 (%ld bytes)\n", (long)alloc_size);
        // Clean up previous allocations
        // Add cleanup code here
        exit(EXIT_FAILURE);
    }
    total_allocated += alloc_size;
    log_debug("Total memory allocated: %ld bytes\n", (long)total_allocated);

    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->w1, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    layer_size = config->dim * config->hidden_dim;
    weights->w2 = (float*)malloc(config->n_layers * layer_size * sizeof(float));
    weights->w3 = (float*)malloc(config->n_layers * layer_size * sizeof(float));

    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->w2, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->w3, config->n_layers * layer_size);
    offset += config->n_layers * layer_size * sizeof(float);

    // final rms norm
    weights->rms_final_weight = (float*)malloc(config->dim * sizeof(float));
    fseek(file, offset, SEEK_SET);
    read_weights_from_file(file, weights->rms_final_weight, config->dim);
    offset += config->dim * sizeof(float);

    // Skip freq_cis_real and freq_cis_imag
    offset += config->seq_len * head_size * sizeof(float);

    // classifier weights
    if (!shared_weights) {
        weights->wcls = (float*)malloc(config->vocab_size * config->dim * sizeof(float));
        fseek(file, offset, SEEK_SET);
        read_weights_from_file(file, weights->wcls, config->vocab_size * config->dim);
    } else {
        weights->wcls = weights->token_embedding_table;
    }

    log_debug("Checkpoint loaded successfully\n");
    fclose(file);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // Replace munmap with free
    if (t->data) { free(t->data); }
    // No need to close fd since we're not using mmap
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    int j;
    for (j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += REAL_EPSILON;
    ss = 1.0f / safe_sqrt(ss);
    // normalize and scale
    for (j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float sum;
    float max_val;
    int i;
    // find max value (for numerical stability)
    max_val = x[0];
    for (i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    sum = 0.0f;
    for (i = 0; i < size; i++) {
        x[i] = safe_exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    int i, j;
    float val;
    size_t offset;

    for (i = 0; i < d; i++) {
        val = 0.0f;
        for (j = 0; j < n; j++) {
            offset = i * n + j;
            val += w[offset] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p;
    TransformerWeights* w;
    RunState* s;
    float *x;
    int dim;
    int kv_dim;
    int kv_mul;
    int hidden_dim;
    int head_size;
    int head_dim;
    float* content_row;
    uint64 l;
    int loff;
    int h;
    int i, j, t;
    float val, freq, fcr, fci, a;
    float* vv;
    float* q;
    float* att;
    float* xb;
    float score;
    long matmul_size;
    float v0, v1;

    // initialize variables
    p = &transformer->config;
    w = &transformer->weights;
    s = &transformer->state;
    x = s->x;
    dim = p->dim;
    kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    kv_mul = p->n_heads / p->n_kv_heads;
    hidden_dim = p->hidden_dim;
    head_size = dim / p->n_heads;

    // copy the token embedding into x
    content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul_size = l*dim*dim;
        matmul(s->q, s->xb, w->wq + matmul_size, dim, dim);

        matmul_size = l*dim*kv_dim;
        matmul(s->k, s->xb, w->wk + matmul_size, dim, kv_dim);

        matmul(s->v, s->xb, w->wv + matmul_size, dim, kv_dim);

        // RoPE relative positional encoding
        for (i = 0; i < dim; i+=2) {
            head_dim = i % head_size;
            freq = 1.0f / pow(10000.0f, head_dim / (float)head_size);
            val = pos * freq;
            fcr = cos(val);
            fci = sin(val);
            if (i < kv_dim) {
                v0 = s->k[i];
                v1 = s->k[i+1];
                s->k[i]   = v0 * fcr - v1 * fci;
                s->k[i+1] = v0 * fci + v1 * fcr;
            }
            v0 = s->q[i];
            v1 = s->q[i+1];
            s->q[i]   = v0 * fcr - v1 * fci;
            s->q[i+1] = v0 * fci + v1 * fcr;
        }

        // multihead attention
        for (h = 0; h < p->n_heads; h++) {
            q = s->q + h * head_size;
            att = s->att + h * p->seq_len;

            for (t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= safe_sqrt(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (t = 0; t <= pos; t++) {
                vv = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                a = att[t];
                for (i = 0; i < head_size; i++) {
                    xb[i] += a * vv[i];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        for (i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        for (i = 0; i < hidden_dim; i++) {
            val = s->hb[i];
            val *= (1.0f / (1.0f + safe_exp(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        for (i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    FILE *file;
    int i, j;
    int len;
    size_t total_allocated = 0;
    size_t vocab_ptr_size;
    size_t vocab_scores_size;
    float score;
    float temp_score;  // Always 4 bytes
    unsigned char len_bytes[4];
    long file_size;
    unsigned char raw_bytes[32];

    log_debug("\nStarting build_tokenizer\n");
    log_debug("Tokenizer path: %s\n", tokenizer_path);
    log_debug("Size of float: %d\n", sizeof(float));
    log_debug("Size of float: %d\n", sizeof(float));
    log_debug("Vocab size: %d\n", vocab_size);

    t->vocab_size = vocab_size;
    vocab_ptr_size = vocab_size * sizeof(char*);
    vocab_scores_size = vocab_size * sizeof(float);
    total_allocated += vocab_ptr_size + vocab_scores_size;

    log_debug("About to allocate vocab arrays:\n");
    log_debug("  - vocab array: %ld bytes\n", (long)vocab_ptr_size);
    log_debug("  - vocab_scores: %ld bytes\n", (long)vocab_scores_size);

    t->vocab = (char**)malloc(vocab_ptr_size);
    t->vocab_scores = (float*)malloc(vocab_scores_size);
    t->sorted_vocab = NULL;

    if (!t->vocab || !t->vocab_scores) {
        log_debug("Failed to allocate vocab arrays\n");
        exit(EXIT_FAILURE);
    }
    log_debug("Allocated initial vocab arrays successfully\n");

    for (i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    file = fopen(tokenizer_path, "rb");
    if (!file) {
        log_debug("couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    // Peek at first 32 bytes
    if (fread(raw_bytes, 1, 32, file) == 32) {
        log_debug("First 32 bytes of file:\n");
        for (i = 0; i < 32; i++) {
            log_debug("%02x ", raw_bytes[i]);
            if ((i + 1) % 8 == 0) log_debug("\n");
        }
    }
    fseek(file, 0, SEEK_SET);

    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    log_debug("Tokenizer file size: %ld bytes\n", file_size);
    log_debug("Opened tokenizer file successfully\n");

    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        log_debug("failed to read max_token_length\n");
        exit(EXIT_FAILURE);
    }
    log_debug("Read max_token_length: %d\n", t->max_token_length);

    for (i = 0; i < vocab_size; i++) {
        // Read score as 4-byte float first
        if (fread(&temp_score, sizeof(float), 1, file) != 1) {
            log_debug("failed to read vocab score at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        // Convert to whatever float is
        score = (float)temp_score;
        t->vocab_scores[i] = score;

        // Read length bytes one at a time
        for (j = 0; j < 4; j++) {
            if (fread(&len_bytes[j], 1, 1, file) != 1) {
                log_debug("failed to read length byte %d at index %d\n", j, i);
                exit(EXIT_FAILURE);
            }
        }

        len = (len_bytes[3] << 24) | (len_bytes[2] << 16) | (len_bytes[1] << 8) | len_bytes[0];

        if (len <= 0 || len > t->max_token_length) {
            log_debug("Invalid token length %d at index %d (max allowed: %d)\n",
                     len, i, t->max_token_length);
            exit(EXIT_FAILURE);
        }

        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i]) {
            log_debug("Failed to allocate token string at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        total_allocated += len + 1;

        if (fread(t->vocab[i], len, 1, file) != 1) {
            log_debug("failed to read token text at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }

    log_debug("Successfully read all tokens and scores\n");
    log_debug("Total tokenizer memory allocated: %ld bytes (%.2f MB)\n",
              (long)total_allocated, (float)total_allocated / (1024*1024));

    fclose(file);
    log_debug("Tokenizer built successfully\n");
}

void free_tokenizer(Tokenizer* t) {
    int i;
    for (i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece;
    unsigned char byte_val;
    piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    unsigned char byte_val;
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok;
    TokenIndex *res;
    tok.str = str;
    tok.id = 0; // id not used for searching
    res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
//     // Mark unused parameters
//     (void)t;
//     (void)text;
//     (void)bos;
//     (void)eos;

//     // Hardcoded test tokens
//     log_debug("Using hardcoded test tokens\n");

//     *n_tokens = 6;
//     tokens[0] = 1;     // BOS
//     tokens[1] = 22172;
//     tokens[2] = 727;
//     tokens[3] = 590;
//     tokens[4] = 1024;
//     tokens[5] = 338;
// }

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    int i;
    char* str_buffer;
    char* c;
    float best_score;
    int best_id;
    int best_idx;
    int id;
    size_t str_len;

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        best_score = -1e10;
        best_id = -1;
        best_idx = -1;

        for (i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    uint64 rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int i;
    int max_i = 0;
    float max_p = probabilities[0];
    for (i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    int i;
    float cumulative_prob;
    int last_idx;
    float r;
    float cdf;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    cumulative_prob = 0.0f;
    last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    r = coin * cumulative_prob;
    cdf = 0.0f;
    for (i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, uint64 rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(uint64 *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1D) >> 32;
}
float random_f32(uint64 *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    int q;
    float coin;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    int* prompt_tokens;
    int num_prompt_tokens;
    long start;
    long end;
    int next;
    int token;
    int pos;
    float* logits;
    char* piece;
    if (prompt == NULL) { prompt = empty_prompt; }

    printf("\nStarting generation:\n");
    printf("Prompt: '%s'\n", prompt);
    printf("Steps: %d\n", steps);
    printf("Transformer config: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d\n",
           transformer->config.dim, transformer->config.hidden_dim,
           transformer->config.n_layers, transformer->config.n_heads);

    // encode the (string) prompt into tokens sequence
    printf("Allocating prompt tokens array...\n");
    num_prompt_tokens = 0;
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    if (!prompt_tokens) {
        printf("Failed to allocate prompt tokens array\n");
        exit(EXIT_FAILURE);
    }

    printf("Encoding prompt...\n");
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    printf("Encoded %d prompt tokens\n", num_prompt_tokens);
    printf("First token: %d\n", prompt_tokens[0]);

    // start the main loop
    start = 0;
    token = prompt_tokens[0];
    pos = 0;
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) {
            printf("Found BOS token, stopping\n");
            break;
        }

        piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) {
            start = time_in_ms();
            printf("Started timing at %ld ms\n", start);
        }
    }
    printf("\n");

    if (pos > 1) {
        end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    printf("Freeing prompt tokens...\n");
    free(prompt_tokens);
    printf("Generation complete\n");
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;
    float* logits;
    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tok512.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    uint64 rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int i;
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif

