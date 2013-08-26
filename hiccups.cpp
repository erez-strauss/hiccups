//
// hiccup.cpp
// Measure system hiccups - how many iterruptions are taking place when an application is running in tight loop.
//
// Future work:
//  Report system information: OS, Kernel, cpus, boot kernel line, uname -a ...)
//  Add intel PMC reporting per thread / per cpu socket
//
// Compile:
//  g++ -pthread -O3 -W -Wall -o hiccups hiccups.cpp -lpthread
//
// Author: Erez Strauss <erez@erezstrauss.com>, 2010 - 2013 (C)
//

#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h> // needed for gettid()
#include <ctype.h>

#define MILLIS_IN_SEC (1000UL)
#define MICROS_IN_SEC (1000000UL)
#define NANOS_IN_SEC  (1000000000UL)

using namespace std;

// Generic utility functions:
#define rdtsc() ({ register uint32_t lo, hi;			\
      __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));	\
      (uint64_t)hi << 32 | lo; })

static inline pid_t gettid() { return syscall(__NR_gettid); }
static void stackreserve(uint32_t sz = 64*1024);
static void runoncpu(int cn);

static inline void* memzalloc(unsigned long sz) { void* p = malloc(sz); memset(p, 0, sz); return p; }
static inline void  memfree(void* p) { free(p); }

#define likely(X)    __builtin_expect((X),1)
#define unlikely(X)  __builtin_expect((X),0)

// Hiccups parameters and data structures

const int MaxCpuCores = 256;		// Maximum cores the app supports, if more exist, change the  masks handling

const int BlockSize = (2*1024*1024);
#define HPGSZ(SZ) (((SZ)/BlockSize+1)*BlockSize)

struct SysCpuInfo {
  //  int     cpu2sock_[MaxCpuCores];
  int     maxcpus_;
  uint	  ccpermicro_;			// CPU Cycles, from /proc/cpuinfo
  uint    ccpermilli_;
  ulong   ccpersec_;          		// calc by avg cc per sec

  void  init();
  void  getisolcpu();
  long  getccpermicro();
};
static SysCpuInfo syscpus;

struct HiccupsInfo;

struct ThrdStart {
  HiccupsInfo*  hudata_;
  pthread_t	thread_desc_;
  int           thrdid_;
  uint64_t      rv_;
};

//  const int MaxHiccup = 250; 	// max length of Hiccup in microseconds
//  int Param_NSperBin = 200; 	// how many nano-seconds per bin, dictated by requested resolution
//  int Param_CCperBin = 0;	// CPU cycles per bin, base on cpu clock rate and NSperBin
struct HiccupConfig {
  int runtime_;			// Total run time, in seconds.
  int resolution_;		// nano-seconds per bin, 200ns, 1/5us.
  int ccperbin_;
  int verbose_;			// report historgrams
  int priority_;		// RT priority
  int policy_;		 	// OTHER, FIFO, Round-Robin
  int bins_;          		// number of bins used by each thread.
  int maxcpus_;
  int maxdelay_us_;		// all delays longer then this are in the last bin.
  int cpus_[MaxCpuCores];  	// test run on these cpu cores.

  HiccupConfig () :
    runtime_(5),		// Default runtime of 5 seconds
    resolution_(200),		// Default histogram resolution of 200 nano seconds
    ccperbin_(0),		// CpuCycles (timestamp counter) per histogram bin
    verbose_(0),
    priority_(0),		// Priority
    policy_(SCHED_OTHER),	// Scheduler policy
    bins_(-1), 			// calc at init, before allocating the bins: MaxHiccup * CCperUS / ccperbin_
    maxcpus_(0),
    maxdelay_us_(200)		// all delay longer then 200 mico-seconds are collected to the last bin
  { }
  
  void init() {
    bins_ = maxdelay_us_ * syscpus.ccpermicro_ / ( resolution_ * syscpus.ccpermicro_ / 1000 ) + 1;
    ccperbin_ = resolution_ * syscpus.ccpermicro_ / 1000 + 1;
  }
  void print() {
    fprintf(stdout, "Hiccups Configuration:\n"
	    "  Run Time: %d [s]\n"
	    "  Histogram resolution: %d [ns]\n"
	    "  RT Priority: %d\n"
	    "  bins: %d\n"
	    "  cpus(%d):",
	    runtime_, resolution_, priority_, bins_, maxcpus_);
    for (int ii = 0; ii < maxcpus_; ii++)
      fprintf(stdout, " %d", cpus_[ii]);
    fprintf(stdout, "\n");
  }
};

static HiccupConfig conf;

struct HiccupsInfo 	// Hiccup Detection Thread
{
  enum { RSERROR, RSWAIT, RSRUN, RSSTOP, RSEXIT };
  uint64_t run();
  void print_us();
  void print_cc();

  static void* runthread(void* vp);

  uint64_t min_, minidx_,
    max_, maxidx_,
    avg_, samples_,
    runtimecc_, startcc_, endcc_, runtimeus_;
  int id_, cpuid_; // id - thread ID, cpuid_ - CPU core ID
  struct HUbin {
    uint64_t n_, sum_;
  };
  HUbin* bins_; // Hiccup bins, allocated, two
  
  static HiccupsInfo* hudatap_[];
  static volatile int tststate_;
  static int numcores_;
  static volatile int active_;
  static volatile int runstate_;
};

volatile int HiccupsInfo::active_ __attribute__((aligned(0x40)));
volatile int HiccupsInfo::runstate_ __attribute__((aligned(0x40)));
HiccupsInfo* HiccupsInfo::hudatap_[MaxCpuCores];

static void setcpus(const char* cpuspec) {
  uint64_t mask = 0;
  char sep[64];
  int so = 0, x1 = 0, x2 = 0, c, lc = -1;

  while (sscanf(cpuspec + so, "%d%n%[ \t,\n]%n", &c, &x1, sep, &x2) >= 1) {
    if (lc >= 0 && c < 0) {
      c = -c;
      for (int ii = lc; ii < c; ii++)
	mask |= (1UL << ii);
    }
    if (c< 0 || c >= syscpus.maxcpus_) {
      fprintf(stderr, "cpu id out of range %d\n", c);
      exit(1);
    }
    mask |= (1UL << c);
    so += x2 ? x2 : x1;
    lc = c;
    x1 = x2 = 0;
  }
  for (int ii = 0; ii < 64; ii++)
    if (mask & (1UL << ii))
      conf.cpus_[conf.maxcpus_++] = ii;
  fprintf(stdout, "Running on %d cpu%s:\n", conf.maxcpus_, (conf.maxcpus_>1)?"s":"");
  for (int ii = 0; ii < conf.maxcpus_; ii++)
    fprintf(stdout, " %d", conf.cpus_[ii]);
  fprintf(stdout, "\n");
}

void* HiccupsInfo::runthread(void* vp) {
  ThrdStart* ts = (ThrdStart*)vp;
  int id = ts->thrdid_;
  int cpu;

  runoncpu(conf.cpus_[id]);
  if ((cpu = sched_getcpu()) < 0) {
    perror("sched_getcpu");
    exit(1);
  }
  if (cpu != conf.cpus_[id]) {
    fprintf(stderr, "Thread %d: tid: %d, running on wrong cpu: %d, expected: %d\n",
	    id, gettid(), cpu, conf.cpus_[id]);
    exit(1);
  }
  unsigned long sz = HPGSZ(sizeof(HiccupsInfo));

  fprintf(stdout, "thread#: %3d tid: %d size %lu cpu[%d]: %d\n",
	  id, gettid(), sz, id, conf.cpus_[id]);
  HiccupsInfo::hudatap_[id] = ts->hudata_ = (HiccupsInfo*)memzalloc(sz);
  ts->hudata_->cpuid_ = conf.cpus_[id];
  ts->hudata_->id_ = id;
  ts->hudata_->bins_ = (HUbin*)memzalloc(conf.bins_ * sizeof(bins_[0]));

  ts->rv_ = ts->hudata_->run();
  if (id)
    pthread_exit(0);
  return 0;
}

uint64_t HiccupsInfo::run() {
  uint64_t cc, lcc, ccend, dt, tsum = 0, ccstart;
  int b;

  stackreserve();
  for (int ii = 0; ii < conf.bins_; ii++ )
    bins_[ii].n_ = bins_[ii].sum_ = 0;
  if (conf.priority_) {
    sched_param sp;
    memset(&sp, 0, sizeof (sp));
    sp.sched_priority = conf.priority_;
    if (sched_setscheduler(gettid(), conf.policy_, &sp))
      perror("set scheduler");
  }
  if (id_) {
    __sync_add_and_fetch(&active_, 1);
    while (runstate_ != RSRUN)
      ;
  }
  else {
    while (1) {
      int z = active_;
      __sync_synchronize();
      if (z >= conf.maxcpus_ - 1)
        break;
      usleep(2000);
    }
    runstate_ = RSRUN;
  }
  ccend = (lcc = cc = ccstart = startcc_ = rdtsc()) + conf.runtime_ * syscpus.ccpersec_;
  if (id_)
    ccend += 500 * syscpus.ccpermicro_;
  while (runstate_ == RSRUN) {
    // __sync_synchronize();  // memory barrier
    cc = rdtsc();
    if (cc > ccend)
      break;
    dt = cc -lcc;
    tsum += dt;
    if (likely(samples_++ > 0)) {
      if (unlikely (min_ > dt)) {
	min_ = dt;
	minidx_ = samples_ - 1;
      } else if (unlikely(max_ < dt)) {
	max_ = dt;
	maxidx_ = samples_ - 1;
      }
    }
    else {
      min_ = max_ = dt;
      minidx_ = maxidx_ = 0;
    }
    b = dt / conf.ccperbin_;
    if (unlikely(b > conf.bins_))
      b = conf.bins_ - 1;
    bins_[b].n_++;
    bins_[b].sum_ += dt;
    lcc = cc;
  }
  endcc_ = ccend = cc;
  runtimecc_ = endcc_ - startcc_;
  sched_param sp;
  memset(&sp, 0, sizeof(sp));
  if (conf.priority_ && sched_setscheduler(gettid(), SCHED_OTHER, &sp))
    perror("reset priority failed");

  if (id_) {
    __sync_sub_and_fetch(&active_, 1);
  } else {
    runstate_ = RSSTOP;
    while (active_ > 0)
      __sync_synchronize();
    runstate_ = RSEXIT;
  }
  avg_ = samples_ ? (tsum / samples_) : 0;
  if (id_ == 0)
    runstate_ = RSEXIT;

  usleep((cpuid_+1)*3000);
  return ccend - ccstart;
}

void HiccupsInfo::print_us() // print in micro seconds.
{
  fprintf(stdout,
	  "thread#: %d core#: %d samples: %ld  avg: %.4f min: %.4f (@%ld) max: %.4f (@%ld) cycles: %lu start: %lu end: %lu\n",
	  id_, conf.cpus_[id_], samples_, 1.0 * avg_ / syscpus.ccpermicro_,
	  1.0 * min_ / syscpus.ccpermicro_, minidx_,
	  1.0 * max_ / syscpus.ccpermicro_, maxidx_,
	  runtimecc_, startcc_, endcc_);
  if (conf.verbose_) {
    uint64_t dtsum = 0;
    for (int ii = 0; ii < conf.bins_; ii++)
      dtsum += bins_[ii].sum_;
    uint64_t psamples = 0, pdtsum = 0;
    for (int ii = 0; ii < conf.bins_; ii++) {
      if (bins_[ii].n_ == 0)
	continue;
      psamples += bins_[ii].n_;
      pdtsum += bins_[ii].sum_;
      fprintf(stdout,
	      "  [%06.2f-%06.2f): %-14ld %-14ld %11.5f%%(%9.5f)  %8.5f%%(%9.5f)\n",
	      1.0 * ii * conf.ccperbin_ / syscpus.ccpermicro_,
	      1.0 * (ii+1) * conf.ccperbin_ / syscpus.ccpermicro_,
	      bins_[ii].n_, samples_ - psamples,
	      samples_ ? (100.0*bins_[ii].n_/samples_) : 0.0,
	      psamples ? (100.0*psamples/samples_) : 0.0,
	      (dtsum != 0) ? (100.0*bins_[ii].sum_/dtsum) : 0.0,
	      pdtsum ? (100.0*pdtsum/dtsum) : 0.0);
    }
    fprintf(stdout, "\n");
  }
}

void HiccupsInfo::print_cc() // print in cpu cycles
{
}

static const char* usagestr =
  "usage: hiccups [-v] [-b nano] [-c cpus-list] [-t seconds] [-n nice|-r RR|-f FF]\n"
  "nano - nano seconds per histogram bin\n"
  "cpulist - on which cpu cores to run\n"
  "seconds - how long to rung the test\n"
  "nice RR FF - priority and scheduling policy\n";

int main(int argc, char** argv)
{
  syscpus.init();
  
  if (mlockall(MCL_CURRENT | MCL_FUTURE))
    fprintf(stderr, "failed to lock all memory (ignored)\n");
  
  int oc;
  while ((oc = getopt(argc, argv, "hb:vr:t:p:f:c:n:")) != -1) {
    switch (oc) {
    case 'v':
      conf.verbose_++;
      break;
    case 'b': // nano seconds per bin. (10ns - 5000ns).
      conf.verbose_++;
      conf.resolution_ = atoi(optarg);
      break;
    case 't':
      conf.runtime_ = atoi(optarg);
      break;
    case 'n':
      conf.policy_ = SCHED_OTHER;
      conf.priority_ = atoi(optarg);
      break;
    case 'r':
      conf.policy_ = SCHED_RR;
      conf.priority_ = atoi(optarg);
      break;
    case 'f':
      conf.policy_ = SCHED_FIFO;
      conf.priority_ = atoi(optarg);
      break;
    case 'c':
      setcpus(optarg);
      break;
    case 'h':
    default:
      fprintf(stderr, "%s\n", usagestr);
      exit(0);
    }
  }

  conf.init();
  runoncpu(conf.cpus_[0]);
  fprintf(stdout, "main thread on cpu core#: %d\n", conf.cpus_[0]);
  conf.print();
  ThrdStart ta[MaxCpuCores];
  memset(&ta, 0, sizeof (ta));
  for (int ii = 1; ii < conf.maxcpus_; ii++) {
    ta[ii].thrdid_ = ii;
    pthread_create(&ta[ii].thread_desc_, 0, HiccupsInfo::runthread, &ta[ii]);
  }
  HiccupsInfo::runthread(&ta[0]);
  for (int ii = 0; ii < conf.maxcpus_; ii++) {
    ta[ii].hudata_->print_us();
  }
}

static void runoncpu(int cpu)
{
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);

  if (sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset) != 0) {
    fprintf(stderr, "failed to set cpu affinity: %d, errno: %d '%s'\n",
	    cpu, errno, strerror(errno));
    exit(1);
  }
}

static void stackreserve(const uint32_t sz)
{
  volatile char dummy[sz];

  memset((void*)dummy, 0, sz);
}

void SysCpuInfo::getisolcpu()
{
  FILE* fp = fopen("/proc/cmdline", "r");
  char line[1024];
  char *p, *pe;

  while (fgets(line, sizeof(line), fp) != NULL)
    if ((p = strstr(line, "isolcpus=")) != NULL) {
      p += strlen("isolcpus=");
      for (pe = p; isdigit(*pe) || *pe == ',' || *pe == '-'; pe++)
	;
      *pe = '\0';
      setcpus(p);
      fclose(fp);
      return;
    }
}

long SysCpuInfo::getccpermicro()
{
  uint64_t ccstart0, ccstart1, ccend0, ccend1, us, cc;
  timeval  todstart, todend;

  ccstart0 = rdtsc();
  gettimeofday(&todstart, 0);
  ccstart1 = rdtsc();
  usleep(10000);      // sleep for 10 milli seconds
  ccend0 = rdtsc();
  gettimeofday(&todend, 0);
  ccend1 = rdtsc();

  us = ( todend.tv_sec - todstart.tv_sec ) * 1000000UL + todend.tv_usec - todstart.tv_usec;
  cc = (ccend1 + ccend0) / 2 - (ccstart1 + ccstart0) / 2;

  float r = (double)cc * 1000 / us;

  // printf("get cpu clock: cc: %lu  us: %lu\n  cc/mico: %.4f\n  cc/mili: %.4f\n", cc, us, r / 1000.0, r);
  ccpersec_   = r * 1000;
  ccpermilli_ = r;
  ccpermicro_ = r / 1000;
  return r / 1000;
}

void SysCpuInfo::init()
{
  int maxcpus = sysconf(_SC_NPROCESSORS_ONLN);
  long r0, r1, r2;

  maxcpus_ = maxcpus;
  r0 = getccpermicro();
  r1 = getccpermicro();
  r2 = getccpermicro();
  if (r0 != r1 || r1 != r2) {
    fprintf(stderr, "Detected non stable clock rates: %ld %ld %ld\n", r0, r1, r2);
    // exit(1);
  }
}
