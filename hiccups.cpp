//
// hiccup.cpp
// Measure system hiccups - how many iterruptions are taking place when an application
// is running in tight loop.
//
// Future work:
//  Report system information: OS, Kernel, cpus, boot kernel line, uname -a ...)
//  Add intel PMC reporting per thread / per cpu socket
//
// Compile:
//  g++ -std=c++17 -pthread -O3 -W -Wall -o hiccups hiccups.cpp -lpthread
//
// Author: Erez Strauss <erez@erezstrauss.com>, Copyright (C) 2010 - 2020
//

#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>

#include <atomic>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <array>

namespace hiccups {

template<typename T, unsigned = 0>
class SyncAccess
{
public:
    SyncAccess(T& o) : ul_(m_), o_(o) {}
    ~SyncAccess()
    {
        o_.flush();
        ul_.unlock();
    }
    T& operator()() { return o_; }
    T& o() { return o_; }

private:
    std::unique_lock<std::mutex> ul_;
    T&                           o_;
    inline static std::mutex     m_{};
};

template<typename T, unsigned N = 0>
inline SyncAccess<T, N> syncedAccess(T& o)
{
    return SyncAccess<T, N>(o);
}
template<typename S, typename T, unsigned N = 0>
inline S& operator<<(SyncAccess<S, N>&& sa, const T&& d)
{
    return sa.o() << "[" << gettid() << "]: " << d;
}
template<typename S, typename T, unsigned N = 0, size_t DN>
inline S& operator<<(SyncAccess<S, N>&& sa, const T (&d)[DN])
{
    return sa.o() << "[" << gettid() << "]: " << d;
}

inline uint64_t rdtsc() { return __rdtsc(); }
inline void     runoncpu(int cpu);
static inline pid_t gettid() { return syscall(__NR_gettid); }

pid_t mytid()
{
    static thread_local pid_t my_tid{gettid()};
    return my_tid;
}

#define likely(X) __builtin_expect((X), 1)
#define unlikely(X) __builtin_expect((X), 0)

const char usageMsg[]{
    "usage: hiccups [-v] [-b nano] [-c cpus-list] [-t seconds] [-n nice|-r RR|-f "
    "FF]\n"
    "nano - nano seconds per histogram bin\n"
    "cpulist - on which cpu cores to run\n"
    "seconds - how long to rung the test\n"
    "nice RR FF - priority and scheduling policy\n"};
class SysCpuInfo
{
    // 1. cpu rate, 2. maximum number of cores, 3. isolated cpu in the system.
    void     init();
    uint64_t calc_ccpermicro();

public:
    SysCpuInfo() { init(); }

    [[nodiscard]] auto ccpermicro() const noexcept { return ccpermicro_; }
    [[nodiscard]] auto ccpersec() const noexcept { return ccpersec_; }
    [[nodiscard]] auto maxcpus() const noexcept { return maxcpus_; }

    static SysCpuInfo& instance();
    static std::string getisolcpu();

private:
    uint32_t maxcpus_{0};
    uint32_t ccpermicro_{0};
    uint64_t ccpersec_{0};
};

class HiccupsInfo;
class HiccupsDetector;

class ThreadContext
{
public:
    ThreadContext()                     = default;
    ThreadContext(const ThreadContext&) = delete;
    ThreadContext(ThreadContext&&)      = default;
    ThreadContext& operator=(const ThreadContext&) = delete;
    ThreadContext& operator=(ThreadContext&&) = default;
    ~ThreadContext();

    void               setThreadId(unsigned tid) { threadId_ = tid; }
    [[nodiscard]] auto id() const noexcept -> unsigned { return threadId_; }
    void               run(HiccupsDetector& detector);
    void               set_thread(std::thread&& t) { thread_ = std::move(t); }
    std::ostream&      report(std::ostream& os) const;

private:
    std::thread                  thread_{};
    std::unique_ptr<HiccupsInfo> hiccupsInfo_{nullptr};
    unsigned                     threadId_{~0U};
};

std::ostream& operator<<(std::ostream& os, const ThreadContext& tc)
{
    return tc.report(os);
}

enum class RunState { RSINIT, RSWAIT, RSRUN, RSSTOP, RSEXIT };

class HiccupsConfig
{
public:
    int      runtime_{5};       // Total run time, in seconds, default 5 seconds
    int      resolution_{200};  // histogram resolution nano-seconds per bin, 200ns
    unsigned ccperbin_{0};      // CpuCycles per histogram bin, calculated
    int      verbose_{0};       // Report historgrams
    int      priority_{0};      // nice or RT priority levels
    int      policy_{SCHED_OTHER};  // OTHER, FIFO, Round-Robin
    unsigned bins_{0};  // number of bins, calc at init, before allocating the bins
    int      maxdelay_us_{200};  // all delays longer then this are in the last bin.
    std::vector<int> cpus_{};    // test run on these cpu cores.

    HiccupsConfig(int argc, char** argv);

    void          process_args(int argc, char** argv);
    void          setcpus(const char* cpuspec);
    void          init();
    std::ostream& report(std::ostream& os) const;
};

std::ostream& operator<<(std::ostream& os, const HiccupsConfig& hc)
{
    return hc.report(os);
}

class HiccupsInfo  // Hiccup Detection Thread
{
public:
    HiccupsInfo(HiccupsDetector& d, unsigned tid);

    uint64_t      run();
    std::ostream& report(std::ostream& os) const;

private:
    void update_histogram(uint64_t dt);

    HiccupsDetector& detector_;
    uint64_t         min_{0};
    uint64_t         minidx_{0};
    uint64_t         max_{0};
    uint64_t         maxidx_{0};
    uint64_t         avg_{0};
    uint64_t         samples_{0};
    uint64_t         runtimecc_{0};
    uint64_t         startcc_{0};
    uint64_t         endcc_{0};
    uint64_t         ccperbin_{0};
    unsigned         id_{0};     // id - thread ID
    unsigned         cpuid_{0};  // cpuid_ - CPU core ID

    class HUbin
    {
    public:
        uint64_t n_{0}, sum_{0};
    };
    std::vector<HUbin> bins_{};
};

class HiccupsDetector
{
public:
    HiccupsDetector(const HiccupsDetector&) = delete;
    HiccupsDetector(HiccupsDetector&&)      = delete;
    HiccupsDetector& operator=(const HiccupsDetector&) = delete;
    HiccupsDetector& operator=(HiccupsDetector&&) = delete;

    explicit HiccupsDetector(HiccupsConfig& c) : config_(c)
    {
        for (auto& x : runSteps_) x = 0;
    }

    void operator()();

    const HiccupsConfig& config() { return config_; }

    void               activeInc() { ++active_; }
    void               activeDec() { --active_; }
    [[nodiscard]] auto active() volatile noexcept { return active_.load(); }

    [[nodiscard]] RunState state() const noexcept { return run_state_.load(); }
    void                   state(RunState s) { run_state_ = s; }
    void                   waitForAll(unsigned tid);

    void stepTogether(unsigned s) { allTogether(runSteps_[s]); }

private:
    void allTogether(std::atomic<unsigned>& c)
    {
        unsigned g = threads_.size();
        ++c;
        while (c.load() < g)
            ;
    }

    std::array<std::atomic<uint32_t>, 6> runSteps_ __attribute__((aligned(0x40))){0};
    const HiccupsConfig&                 config_;
    volatile std::atomic<RunState>       run_state_
        __attribute__((aligned(0x40))){RunState::RSINIT};
    std::atomic<uint64_t>      active_ __attribute__((aligned(0x40))){0};
    std::vector<ThreadContext> threads_{};
};

HiccupsConfig::HiccupsConfig(int argc, char** argv)
{
    process_args(argc, argv);
    init();
}

void HiccupsConfig::init()
{
    auto tsc_per_micro{SysCpuInfo::instance().ccpermicro()};

    bins_ = maxdelay_us_ * tsc_per_micro / (resolution_ * tsc_per_micro / 1000) + 1;
    ccperbin_ = resolution_ * tsc_per_micro / 1000 + 1;
    if (cpus_.empty())
    {
        auto cpuspec = SysCpuInfo::getisolcpu();
        if (!cpuspec.empty()) setcpus(cpuspec.c_str());
    }
    if (cpus_.empty())
    {
        std::cerr << "Error: could not find isolcpus=.. and no cpus '-c' defined on "
                     "command line\n"
                  << usageMsg << std::endl;
        exit(1);
    }
}

void HiccupsConfig::setcpus(const char* cpuspec)
{
    uint64_t mask = 0;
    char     sep[64];
    int      so = 0, x1 = 0, x2 = 0, c, lc = -1;

    while (sscanf(cpuspec + so, "%d%n%[ \t,\n]%n", &c, &x1, sep, &x2) >= 1)
    {
        if (lc >= 0 && c < 0)
        {
            c = -c;
            for (unsigned ii = lc; ii < unsigned(c); ii++) mask |= (1UL << ii);
        }
        if (c < 0 || c >= int(SysCpuInfo::instance().maxcpus()))
        {
            std::cerr << "Error: cpu id out of range: " << c << std::endl;
            exit(1);
        }
        mask |= (1UL << c);
        so += x2 ? x2 : x1;
        lc = c;
        x1 = x2 = 0;
    }
    for (unsigned ii = 0; ii < 64; ii++)
        if (mask & (1UL << ii)) cpus_.push_back(ii);
    std::cout << "Running on " << cpus_.size() << " cpu"
              << ((cpus_.size() > 1) ? "s" : "") << std::endl;
    for (auto cpu : cpus_) std::cout << " " << cpu;
    std::cout << std::endl;
}

std::ostream& HiccupsConfig::report(std::ostream& os) const
{
    os << "Hiccups Configuration:\n"
          "  Run Time: "
       << runtime_
       << "[s]\n"
          "  Histogram resolution: "
       << resolution_
       << "[ns]\n"
          "  "
       << (policy_ == SCHED_OTHER
               ? "NICE"
               : (policy_ == SCHED_RR
                      ? "RT-ROUND-ROBIN"
                      : (policy_ == SCHED_FIFO ? "RT-FIFO" : "UNKNOWN")))
       << " Priority: " << priority_
       << "\n"
          "  bins: "
       << bins_
       << "\n"
          "  cpus("
       << cpus_.size() << "):";
    for (auto c : cpus_) os << " " << c;
    os << '\n';
    return os;
}

ThreadContext::~ThreadContext()
{
    if (thread_.joinable())
    {
        thread_.join();
    }
}

std::ostream& ThreadContext::report(std::ostream& os) const
{
    return hiccupsInfo_->report(os);
}

void ThreadContext::run(HiccupsDetector& detector)
{
    auto tid = id();
    int  cpu;

    runoncpu(detector.config().cpus_[tid]);
    if ((cpu = sched_getcpu()) < 0)
    {
        perror("sched_getcpu");
        exit(1);
    }
    if (cpu != detector.config().cpus_[tid])
    {
        std::cerr << "Error: Thread " << tid << ": tid: " << gettid()
                  << ", running on wrong cpu: " << cpu
                  << ", expected: " << detector.config().cpus_[tid] << std::endl;
        exit(1);
    }

    syncedAccess(std::cout) << "thread#: " << tid << " cpus[" << tid
                            << "]: " << detector.config().cpus_[tid] << std::endl;

    hiccupsInfo_ = std::make_unique<HiccupsInfo>(detector, tid);
    hiccupsInfo_->run();
}

HiccupsInfo::HiccupsInfo(HiccupsDetector& d, unsigned tid)
    : detector_(d),
      ccperbin_(detector_.config().ccperbin_),
      id_(tid),
      cpuid_(detector_.config().cpus_[tid])
{
    bins_.resize(detector_.config().bins_);
}

inline void HiccupsInfo::update_histogram(uint64_t dt)
{
    uint64_t b{dt / ccperbin_};
    if (unlikely(b >= bins_.size())) b = bins_.size() - 1;
    bins_[b].n_++;
    bins_[b].sum_ += dt;
}

uint64_t HiccupsInfo::run()
{
    uint64_t cc{0}, lcc{0}, ccend{0}, dt{0}, tsum{0}, ccstart{0};

    for (auto& b : bins_) b.n_ = b.sum_ = 0;

    detector_.stepTogether(0);

    detector_.waitForAll(id_);

    if (detector_.config().priority_)
    {
        sched_param sp{};
        memset(&sp, 0, sizeof(sp));
        sp.sched_priority = detector_.config().priority_;
        if (detector_.config().policy_ == SCHED_OTHER)
        {
            if (0 != setpriority(PRIO_PROCESS, 0, detector_.config().priority_))
                perror("failed to setpriority");
        }
        else if (sched_setscheduler(gettid(), detector_.config().policy_, &sp))
            perror("set scheduler");
    }

    detector_.stepTogether(1);

    ccend = (lcc = cc = ccstart = startcc_ = rdtsc()) +
            detector_.config().runtime_ * SysCpuInfo::instance().ccpersec();
    if (id_) ccend += 500 * SysCpuInfo::instance().ccpermicro();
    while (detector_.state() == RunState::RSRUN)
    {
        cc = rdtsc();
        if (cc > ccend) break;
        dt = cc - lcc;
        tsum += dt;
        if (likely(samples_++ > 0))
        {
            if (unlikely(min_ > dt))
            {
                min_    = dt;
                minidx_ = samples_ - 1;
            }
            else if (unlikely(max_ < dt))
            {
                max_    = dt;
                maxidx_ = samples_ - 1;
            }
        }
        else
        {
            min_ = max_ = dt;
            minidx_ = maxidx_ = 0;
        }
        update_histogram(dt);
        lcc = cc;
    }
    endcc_ = ccend = cc;
    runtimecc_     = endcc_ - startcc_;
    avg_           = samples_ ? (tsum / samples_) : 0;

    detector_.activeDec();

    if (!id_) detector_.state(RunState::RSSTOP);

    detector_.stepTogether(2);

    // while (detector_.active() > 0)
    //    rdtsc();

    sched_param sp{};
    memset(&sp, '\0', sizeof(sp));
    if (detector_.config().priority_ &&
        sched_setscheduler(gettid(), SCHED_OTHER, &sp))
        perror("reset priority failed");

    detector_.stepTogether(3);

    syncedAccess(std::cout) << "thread#: " << id_ << " cpu: " << cpuid_ << ": "
                            << " done" << std::endl;

    detector_.stepTogether(4);

    if (!id_)
    {
        detector_.state(RunState::RSEXIT);
    }

    return ccend - ccstart;
}

std::ostream& HiccupsInfo::report(std::ostream& os) const
{
    const uint64_t tsc_per_micro{SysCpuInfo::instance().ccpermicro()};
    auto&          conf{detector_.config()};
    os.precision(4);
    os << std::fixed << "thread#: " << id_ << " core#: " << cpuid_
       << " samples: " << samples_ << " avg: " << (1.0 * avg_ / tsc_per_micro)
       << " min: " << (1.0 * min_ / tsc_per_micro) << " (@" << minidx_ << ")"
       << " max: " << (1.0 * max_ / tsc_per_micro) << " (@" << maxidx_ << ")"
       << " cycles: " << runtimecc_ << " start: " << startcc_ << " end: " << endcc_
       << std::endl;
    if (detector_.config().verbose_)
    {
        uint64_t dtsum = 0;
        for (auto& b : bins_) dtsum += b.sum_;
        uint64_t psamples = 0, pdtsum = 0;
        for (auto& b : bins_)
        {
            if (b.n_ == 0) continue;

            int64_t ii{(&b - &bins_[0])};

            psamples += b.n_;
            pdtsum += b.sum_;
            os << "  [" << std::setfill('0') << std::right << std::setw(6)
               << std::setprecision(2) << (1.0 * ii * conf.ccperbin_ / tsc_per_micro)
               << "-" << std::setfill('0') << std::right << std::setw(6)
               << std::setprecision(2) << (1.0 * (ii + 1) * ccperbin_ / tsc_per_micro)
               << "): " << std::setfill(' ') << std::left << std::setw(14) << b.n_
               << " " << std::setfill(' ') << std::left << std::setw(14)
               << (samples_ - psamples) << " " << std::setfill(' ') << std::right
               << std::setw(11) << std::setprecision(5)
               << (samples_ ? (100.0 * b.n_ / samples_) : 0.0) << "%("
               << std::setfill(' ') << std::right << std::setw(9)
               << std::setprecision(5)
               << (psamples ? (100.0 * psamples / samples_) : 0.0) << ")  "
               << std::setfill(' ') << std::right << std::setw(9)
               << std::setprecision(5)
               << ((dtsum != 0) ? (100.0 * b.sum_ / dtsum) : 0.0) << "%("
               << std::setfill(' ') << std::right << std::setw(9)
               << std::setprecision(5) << (pdtsum ? (100.0 * pdtsum / dtsum) : 0.0)
               << ")" << std::endl;
        }
        os << std::endl;
    }
    return os;
}

void HiccupsConfig::process_args(int argc, char** argv)
{
    int oc;
    while ((oc = getopt(argc, argv, "hlb:vr:t:p:f:c:n:")) != -1)
    {
        switch (oc)
        {
            case 'v':
                verbose_++;
                break;
            case 'b':  // nano seconds per bin. (10ns - 5000ns).
                verbose_++;
                resolution_ = atoi(optarg);
                break;
            case 't':
                runtime_ = std::strtol(optarg, nullptr, 10);
                break;
            case 'n':
                policy_   = SCHED_OTHER;
                priority_ = atoi(optarg);
                break;
            case 'r':
                policy_   = SCHED_RR;
                priority_ = atoi(optarg);
                break;
            case 'f':
                policy_   = SCHED_FIFO;
                priority_ = atoi(optarg);
                break;
            case 'c':
                setcpus(optarg);
                break;
            case 'l':
                if (mlockall(MCL_CURRENT | MCL_FUTURE))
                    std::cerr << "Error: failed to lock all memory (ignored)"
                              << std::endl;
                break;
            case 'h':
            default:
                std::cerr << usageMsg << std::endl;
                exit(0);
        }
    }
}

void HiccupsDetector::operator()()
{
    runoncpu(config_.cpus_[0]);
    std::cout << "main thread on cpu core#: " << config_.cpus_[0] << "\n " << config_;

    threads_.resize(config_.cpus_.size());

    for (auto& t : threads_) t.setThreadId(&t - &threads_[0]);

    for (auto itr = threads_.begin() + 1; itr != threads_.end(); ++itr)
        itr->set_thread(std::thread{[itr, this]() { itr->run(*this); }});

    threads_[0].run(*this);

    for (auto& t : threads_) std::cout << t;
}

void HiccupsDetector::waitForAll(unsigned tid)
{
    activeInc();
    if (tid == 0)
    {
        while (active() < threads_.size())
            ;
        state(RunState::RSRUN);
    }
    else
    {
        while (state() != RunState::RSRUN)
            ;
    }
}

void runoncpu(int cpu)
{
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    if (sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset) != 0)
    {
        std::cerr << "Error: failed to set cpu affinity: " << cpu
                  << ", errno: " << errno << "'" << strerror(errno) << "'"
                  << std::endl;
        exit(1);
    }
}

SysCpuInfo& SysCpuInfo::instance()
{
    static std::unique_ptr<SysCpuInfo> the_one = std::make_unique<SysCpuInfo>();
    return *the_one;
}

std::string SysCpuInfo::getisolcpu()
{
    const char *p, *pe;

    std::ifstream kcmdlinefile("/proc/cmdline");
    std::string   line;
    while (std::getline(kcmdlinefile, line))
    {
        if ((p = strstr(line.c_str(), "isolcpus=")) != nullptr)
        {
            p += strlen("isolcpus=");
            for (pe = p; isdigit(*pe) || *pe == ',' || *pe == '-'; ++pe)
                ;
            return std::string{p, uint64_t(pe - p)};
        }
    }
    return "";
}

uint64_t SysCpuInfo::calc_ccpermicro()
{
    timeval todStart{};
    timeval todEnd{};

    uint64_t ccstart0{rdtsc()};
    gettimeofday(&todStart, nullptr);
    uint64_t ccstart1{rdtsc()};
    usleep(10000);  // sleep for 10 milli seconds
    uint64_t ccend0{rdtsc()};
    gettimeofday(&todEnd, nullptr);
    uint64_t ccend1{rdtsc()};

    uint64_t us{(todEnd.tv_sec - todStart.tv_sec) * 1000000UL + todEnd.tv_usec -
                todStart.tv_usec};
    uint64_t cc{(ccend1 + ccend0) / 2 - (ccstart1 + ccstart0) / 2};

    double r = (double)cc * 1000 / us;

    ccpersec_   = r * 1000;
    ccpermicro_ = r / 1000;
    return r / 1000;
}

void SysCpuInfo::init()
{
    long maxcpus{sysconf(_SC_NPROCESSORS_ONLN)};
    if (maxcpus > 0) maxcpus_ = maxcpus;

    auto r0{calc_ccpermicro()};
    auto r1{calc_ccpermicro()};
    auto r2{calc_ccpermicro()};
    if (r0 != r1 || r1 != r2)
    {
        std::cerr << "Warning: detected non stable clock rates: " << r0 << " " << r1
                  << " " << r2 << std::endl;
    }
}

}  // namespace hiccups

int main(int argc, char** argv)
{
    hiccups::HiccupsConfig conf{argc, argv};

    hiccups::HiccupsDetector detector(conf);

    detector();

    return 0;
}
