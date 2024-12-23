#ifndef _WIN_H_
#define _WIN_H_

#define WINVER 0x0400        // Target Windows 95/98
#define _WIN32_WINDOWS 0x0400
#define _WIN32_WINNT 0x0400
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <time.h>
#include <stdio.h>

// Add POSIX time types for older Windows
typedef enum {
  CLOCK_REALTIME = 0
} clockid_t;

struct timespec {
  time_t tv_sec;
  long   tv_nsec;
};

// Add after the timespec struct (around line 20)
int clock_gettime(clockid_t clk_id, struct timespec *tp);

// Define fixed-width types compatible with older Windows
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
#if defined(__BORLANDC__)
  typedef __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#else
  typedef signed long long int64_t;
  typedef unsigned long long uint64_t;
#endif

// Use 32-bit compatible ssize_t
#ifndef _SSIZE_T_DEFINED
typedef long ssize_t;
#define _SSIZE_T_DEFINED
#endif

// mman-win32 definitions
#define PROT_NONE       0x0
#define PROT_READ       0x1
#define PROT_WRITE      0x2
#define PROT_EXEC       0x4

#define MAP_FILE        0x0
#define MAP_SHARED      0x1
#define MAP_PRIVATE     0x2
#define MAP_ANONYMOUS   0x20
#define MAP_ANON        MAP_ANONYMOUS
#define MAP_FAILED      ((void *) -1)
#define MAP_FIXED       0x10

// Function declarations
void* mmap(void *addr, size_t len, int prot, int flags, int fildes, ssize_t off);
int munmap(void *addr, size_t len);
int mprotect(void *addr, size_t len, int prot);
int msync(void *addr, size_t len, int flags);
int mlock(const void *addr, size_t len);
int munlock(const void *addr, size_t len);

// Memory management helpers for older Windows
#if defined(__BORLANDC__)
#define HEAP_GRANULARITY (16 * 1024)    // 16KB chunks
#define MAX_ALLOC_SIZE (4 * 1024 * 1024)  // 4MB max single allocation
#define MAX_TOTAL_ALLOC (96 * 1024 * 1024)  // 96MB max total allocation

static size_t total_allocated = 0;

void* safe_malloc(size_t size) {
  void* ptr;
  size_t aligned_size;
  HANDLE heap;
  float size_mb;
  DWORD last_error;
  
  // Debug output for large allocations
  if (size > 1024*1024) {
    size_mb = (float)size / (1024*1024);
    fprintf(stderr, "Large allocation requested: %.2f MB\n", size_mb);
  }
  
  // Check if allocation would exceed our total memory limit
  if (total_allocated + size > MAX_TOTAL_ALLOC) {
    fprintf(stderr, "Memory limit exceeded! Need %.2f MB more\n", 
            (float)(total_allocated + size - MAX_TOTAL_ALLOC) / (1024*1024));
    return NULL;
  }

  // For large allocations, use VirtualAlloc directly
  if (size > MAX_ALLOC_SIZE) {
    ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (ptr) {
      total_allocated += size;
      // Zero the memory since calloc expects this
      memset(ptr, 0, size);
    }
    return ptr;
  }

  // For smaller allocations use heap
  aligned_size = (size + HEAP_GRANULARITY - 1) & ~(HEAP_GRANULARITY - 1);
  heap = GetProcessHeap();
  ptr = HeapAlloc(heap, HEAP_ZERO_MEMORY, aligned_size);

  if (ptr) {
    total_allocated += size;
  } else {
    last_error = GetLastError();
    fprintf(stderr, "HeapAlloc failed: error %lu\n", (unsigned long)last_error);
  }
  return ptr;
}

void safe_free(void* ptr) {
  if (ptr) {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(ptr, &mbi, sizeof(mbi)) == sizeof(mbi) && 
        mbi.State == MEM_COMMIT && 
        mbi.Type == MEM_PRIVATE) {
      total_allocated -= mbi.RegionSize;
      VirtualFree(ptr, 0, MEM_RELEASE);
    } else {
      HANDLE heap = GetProcessHeap();
      // Get the allocation size before freeing
      size_t size = HeapSize(heap, 0, ptr);
      if (size != (size_t)-1) {
        total_allocated -= size;
      }
      HeapFree(heap, 0, ptr);
    }
  }
}

#define malloc(x) safe_malloc(x)
#define free(x) safe_free(x)
#define calloc(n,x) safe_malloc((n)*(x))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* _WIN_H_ */
