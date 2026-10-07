// Unit + acceptance tests for the dynamic DDR expert cache (include/expert_cache.hpp, include/offload_store.hpp).
//
// Build/run: tests/run_components.sh does this; by hand from the repo root:
//   c++ -std=c++20 -O2 -pthread -Iinclude -o /tmp/t tests/test_expert_dram_cache.cpp && /tmp/t
//
// Temp files go to $MOESTREAM_TEST_DIR, else /var/tmp (a real filesystem, so O_DIRECT is exercised; /tmp is tmpfs on
// many systems and refuses O_DIRECT, which only tests the buffered fallback).
#include "offload_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <list>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace moestream::offload;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                                                   \
  do {                                                                                                \
    ++g_checks;                                                                                       \
    if (!(cond)) { ++g_failed; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
  } while (0)

// ------------------------------------------------------------------------------------------ reference model
// Straight transcription of the simulator's LRU (an ordered dict + pinned set), used as the oracle for random tests.
struct RefLru {
  size_t cap;
  std::list<ExpertKey> order; // front = oldest
  std::set<ExpertKey> pinned;
  bool has(const ExpertKey &k) const { return std::find(order.begin(), order.end(), k) != order.end(); }
  void seed(const std::vector<ExpertKey> &hottest, double pin_frac) {
    std::vector<ExpertKey> take(hottest.begin(), hottest.begin() + static_cast<long>(std::min(cap, hottest.size())));
    size_t pins = std::min(take.size(), static_cast<size_t>(static_cast<double>(cap) * pin_frac));
    for (size_t i = 0; i < pins; ++i) pinned.insert(take[i]);
    for (auto it = take.rbegin(); it != take.rend(); ++it) order.push_back(*it); // hottest ends most recent
  }
  struct R { bool hit, evicted; ExpertKey victim; };
  R access(const ExpertKey &k) {
    auto it = std::find(order.begin(), order.end(), k);
    if (it != order.end()) { order.erase(it); order.push_back(k); return {true, false, {}}; }
    R r{false, false, {}};
    if (order.size() >= cap) {
      for (auto o = order.begin(); o != order.end(); ++o)
        if (!pinned.count(*o)) { r.victim = *o; r.evicted = true; order.erase(o); break; }
    }
    order.push_back(k);
    return r;
  }
};

static std::string test_dir() {
  const char *e = std::getenv("MOESTREAM_TEST_DIR");
  std::string base = e && *e ? e : (::access("/var/tmp", W_OK) == 0 ? "/var/tmp" : "/tmp");
  std::string t = base + "/moestream-cache-test.XXXXXX";
  std::vector<char> b(t.begin(), t.end());
  b.push_back('\0');
  if (!::mkdtemp(b.data())) { std::perror("mkdtemp"); std::exit(2); }
  return b.data();
}

// ------------------------------------------------------------------------------------------ cache bookkeeping
static std::vector<ExpertKey> all_keys(int first, int nl, int ne) {
  std::vector<ExpertKey> v;
  for (int l = 0; l < nl; ++l) for (int e = 0; e < ne; ++e) v.push_back({first + l, e});
  return v;
}

static void test_random_equivalence() {
  std::mt19937 rng(12345);
  const size_t caps[] = {1, 2, 5, 17, 40, 60, 100};
  const double pins[] = {0.0, 0.3, 0.5};
  for (size_t cap : caps) for (double pf : pins) for (int seed_mode = 0; seed_mode < 3; ++seed_mode) {
    const int first = 3, nl = 3, ne = 20;
    // pins must leave at least one evictable slot, otherwise the reference (which over-fills) and the cache (which
    // refuses to cache the miss) legitimately differ.
    if (static_cast<size_t>(static_cast<double>(cap) * pf) >= cap) continue;
    ExpertCache::Config cfg{first, nl, ne, cap, ExpertCache::Policy::Lru, pf};
    ExpertCache cache(cfg);
    RefLru ref{cap, {}, {}};
    std::vector<ExpertKey> keys = all_keys(first, nl, ne), hot = keys;
    std::shuffle(hot.begin(), hot.end(), rng);
    size_t n_seed = seed_mode == 0 ? 0 : seed_mode == 1 ? cap / 2 : cap;
    hot.resize(std::min(n_seed, hot.size()));
    cache.seed(hot);
    ref.seed(hot, pf);
    CHECK(cache.size() == ref.order.size());
    std::geometric_distribution<int> skew(0.04);
    for (int step = 0; step < 4000; ++step) {
      ExpertKey k = keys[static_cast<size_t>(std::min<int>(skew(rng), static_cast<int>(keys.size()) - 1))];
      if (rng() % 7 == 0) k = keys[rng() % keys.size()];
      auto a = cache.access(k);
      auto r = ref.access(k);
      if (a.hit != r.hit || a.evicted != r.evicted || (a.evicted && !(a.victim == r.victim))) {
        CHECK(false);
        std::fprintf(stderr, "  diverged: cap=%zu pf=%.1f seed=%d step=%d\n", cap, pf, seed_mode, step);
        break;
      }
      if (step % 97 == 0) {
        CHECK(cache.size() == ref.order.size());
        for (const ExpertKey &q : keys) {
          const bool in_ref = ref.has(q);
          CHECK(cache.resident(q) == in_ref);
          const uint64_t *b = cache.layer_bitmap(q.layer);
          CHECK(((b[q.expert >> 6] >> (q.expert & 63)) & 1ull) == (in_ref ? 1ull : 0ull));
        }
      }
    }
  }
}

static void test_edge_cases() {
  { // capacity 0: nothing is ever cached; every access is a bypassed miss
    ExpertCache c({0, 1, 8, 0, ExpertCache::Policy::Lru, 0.0});
    auto a = c.access({0, 3});
    CHECK(!a.hit && !a.inserted && c.size() == 0 && c.stats().bypass == 1);
  }
  { // all pinned: a full cache has nothing to evict
    ExpertCache c({0, 1, 8, 2, ExpertCache::Policy::Lru, 1.0});
    c.seed({{0, 1}, {0, 2}, {0, 3}});
    CHECK(c.size() == 2 && c.pinned({0, 1}) && c.pinned({0, 2}) && !c.resident({0, 3}));
    auto a = c.access({0, 5});
    CHECK(!a.hit && !a.inserted && !a.evicted && c.stats().bypass == 1 && c.resident({0, 1}) && c.resident({0, 2}));
  }
  { // seed: truncation to capacity, duplicates and out-of-range keys skipped, hotter = more recent, pins never evicted
    ExpertCache c({10, 2, 8, 4, ExpertCache::Policy::Lru, 0.5});
    c.seed({{10, 0}, {10, 0}, {99, 1}, {10, 1}, {11, 2}, {11, 3}, {11, 4}, {11, 5}});
    CHECK(c.size() == 4);
    CHECK(c.pinned({10, 0}) && c.pinned({10, 1}) && !c.pinned({11, 2}));
    auto order = c.lru_order(); // least recent first: the colder unpinned entry comes first
    CHECK(order.size() == 2 && order[0] == ExpertKey({11, 3}) && order[1] == ExpertKey({11, 2}));
    for (int e = 0; e < 8; ++e) c.access({11, e});
    CHECK(c.resident({10, 0}) && c.resident({10, 1}) && c.size() == 4);
  }
  { // static policy: no state change on a miss, hit/miss only
    ExpertCache c({0, 1, 16, 3, ExpertCache::Policy::Static, 0.0});
    c.seed({{0, 4}, {0, 5}, {0, 6}, {0, 7}});
    CHECK(c.size() == 3 && c.resident({0, 6}) && !c.resident({0, 7}));
    CHECK(c.access({0, 4}).hit && !c.access({0, 9}).hit && !c.resident({0, 9}) && c.size() == 3);
    CHECK(c.stats().hits == 1 && c.stats().misses == 1 && c.stats().evictions == 0);
  }
  { // erase keeps list, bitmap and size consistent (pinned and unpinned)
    ExpertCache c({0, 1, 8, 4, ExpertCache::Policy::Lru, 0.25});
    c.seed({{0, 0}, {0, 1}, {0, 2}, {0, 3}});
    c.erase({0, 0}); c.erase({0, 2}); c.erase({0, 7});
    CHECK(c.size() == 2 && !c.resident({0, 0}) && c.resident({0, 1}) && !c.resident({0, 2}));
    auto a = c.access({0, 6});
    CHECK(a.inserted && !a.evicted && c.size() == 3);
  }
  { // out-of-range access is ignored
    ExpertCache c({5, 2, 8, 4, ExpertCache::Policy::Lru, 0.0});
    auto a = c.access({4, 0}), b = c.access({5, 8});
    CHECK(!a.hit && !a.inserted && !b.hit && !b.inserted && c.stats().misses == 0);
    CHECK(c.layer_bitmap(4) == nullptr);
  }
}

static void test_bitmap_all_resident() {
  ExpertCache c({3, 2, 130, 200, ExpertCache::Policy::Lru, 0.0});
  CHECK(c.words_per_layer() == 3);
  c.access({3, 0}); c.access({3, 64}); c.access({3, 129}); c.access({4, 5});
  const uint64_t *b = c.layer_bitmap(3);
  CHECK(b[0] == 1ull && b[1] == 1ull && b[2] == (1ull << 1));
  const int32_t yes[] = {129, -1, 0, 64, -1}, no[] = {129, 1};
  CHECK(c.all_resident(3, yes, 5) && !c.all_resident(3, no, 2) && c.all_resident(3, nullptr, 0));
  const int32_t oob[] = {130};
  CHECK(!c.all_resident(3, oob, 1) && !c.all_resident(9, yes, 1));
}

static void test_ranking(const std::string &dir) {
  const std::string p = dir + "/usage.txt";
  { std::ofstream o(p); o << "11 3 5\n11 4 5\n12 0 9\n11 3 1\n40 1 100\n12 2 0\n"; }
  auto r = rank_hottest_first(p, 11, 2, 6); // layers 11-12 x 6 experts; layer 40 is out of range
  CHECK(r.size() == 12);
  CHECK(r[0] == ExpertKey({12, 0}));         // 9
  CHECK(r[1] == ExpertKey({11, 3}));         // 5+1 = 6
  CHECK(r[2] == ExpertKey({11, 4}));         // 5
  CHECK(r[3] == ExpertKey({11, 0}));         // zeros: (layer, expert) ascending
  CHECK(r[4] == ExpertKey({11, 1}) && r[5] == ExpertKey({11, 2}) && r[6] == ExpertKey({11, 5}) && r[7] == ExpertKey({12, 1}));
}

// ------------------------------------------------------------------------------------------ synthetic GGUF files
static uint8_t pat(int L, int E, int k, size_t j) { return static_cast<uint8_t>((L * 131 + E * 17 + k * 29 + static_cast<int>(j) * 7) & 0xff); }

struct Wr {
  std::ofstream o;
  explicit Wr(const std::string &p) : o(p, std::ios::binary | std::ios::trunc) {}
  template <class T> void put(T v) { o.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
  void str(const std::string &s) { put<uint64_t>(s.size()); o.write(s.data(), static_cast<std::streamsize>(s.size())); }
  void pad_to(uint64_t pos, uint64_t align) { uint64_t cur = static_cast<uint64_t>(o.tellp()); uint64_t t = (cur + align - 1) / align * align; std::vector<char> z(t - cur, 0); o.write(z.data(), static_cast<std::streamsize>(z.size())); (void)pos; }
};

// F32 planes: ne0 = 1024 (a 4096-byte row), ne1 = plane_bytes / 4096 rows.
static std::vector<uint64_t> plane_dims(uint64_t plane) { return {1024, plane / 4096}; }

static void write_store(const std::string &path, const std::vector<ExpertKey> &experts, const std::map<int, uint64_t> &plane_of_layer, bool break_tightness = false) {
  Wr w(path);
  w.o.write("GGUF", 4); w.put<uint32_t>(3); w.put<uint64_t>(experts.size() * 3); w.put<uint64_t>(4);
  w.str("general.alignment"); w.put<uint32_t>(4); w.put<uint32_t>(4096);
  w.str("general.architecture"); w.put<uint32_t>(8); w.str("moestream-offload");
  for (int which = 0; which < 2; ++which) {
    w.str(which == 0 ? "moestream.offload.layer" : "moestream.offload.expert"); w.put<uint32_t>(9); w.put<uint32_t>(4); w.put<uint64_t>(experts.size());
    for (const ExpertKey &k : experts) w.put<uint32_t>(static_cast<uint32_t>(which == 0 ? k.layer : k.expert));
  }
  uint64_t off = 0;
  struct T { std::string name; uint64_t plane, off; ExpertKey k; int kind; };
  std::vector<T> ts;
  for (const ExpertKey &k : experts) {
    const uint64_t pl = plane_of_layer.at(k.layer);
    off = (off + 4095) / 4096 * 4096;
    const char *kinds[3] = {"gate", "up", "down"};
    for (int kind = 0; kind < 3; ++kind) {
      if (break_tightness && kind == 2) off += 4096; // a gap before the down plane
      ts.push_back({"offload.blk." + std::to_string(k.layer) + ".ffn_" + kinds[kind] + "." + std::to_string(k.expert) + ".weight", pl, off, k, kind});
      off += pl;
    }
  }
  for (const T &t : ts) {
    w.str(t.name); auto d = plane_dims(t.plane); w.put<uint32_t>(2); w.put<uint64_t>(d[0]); w.put<uint64_t>(d[1]); w.put<uint32_t>(0); w.put<uint64_t>(t.off);
  }
  w.pad_to(0, 4096);
  const uint64_t data_start = static_cast<uint64_t>(w.o.tellp());
  uint64_t written = 0;
  for (const T &t : ts) {
    std::vector<char> z(t.off - written, 0); w.o.write(z.data(), static_cast<std::streamsize>(z.size()));
    std::vector<char> b(t.plane);
    for (size_t j = 0; j < b.size(); ++j) b[j] = static_cast<char>(pat(t.k.layer, t.k.expert, t.kind, j));
    w.o.write(b.data(), static_cast<std::streamsize>(b.size()));
    written = t.off + t.plane;
  }
  w.pad_to(0, 4096);
  (void)data_start;
}

// Packed main model: blk.L.ffn_{gate,up,down}_exps.weight, dims [1024, rows, n_experts], F32, data aligned to 32.
static void write_model(const std::string &path, const std::vector<int> &layers, int n_experts, const std::map<int, uint64_t> &plane_of_layer) {
  Wr w(path);
  w.o.write("GGUF", 4); w.put<uint32_t>(3); w.put<uint64_t>(layers.size() * 3); w.put<uint64_t>(1);
  w.str("general.alignment"); w.put<uint32_t>(4); w.put<uint32_t>(32);
  uint64_t off = 0;
  struct T { std::string name; uint64_t plane, off; int L, kind; };
  std::vector<T> ts;
  for (int L : layers) for (int kind = 0; kind < 3; ++kind) {
    const char *kinds[3] = {"gate", "up", "down"};
    off = (off + 31) / 32 * 32;
    ts.push_back({"blk." + std::to_string(L) + ".ffn_" + kinds[kind] + "_exps.weight", plane_of_layer.at(L), off, L, kind});
    off += plane_of_layer.at(L) * static_cast<uint64_t>(n_experts);
  }
  for (const T &t : ts) {
    w.str(t.name); auto d = plane_dims(t.plane); w.put<uint32_t>(3); w.put<uint64_t>(d[0]); w.put<uint64_t>(d[1]); w.put<uint64_t>(static_cast<uint64_t>(n_experts)); w.put<uint32_t>(0); w.put<uint64_t>(t.off);
  }
  w.pad_to(0, 32);
  uint64_t written = 0;
  for (const T &t : ts) {
    std::vector<char> z(t.off - written, 0); w.o.write(z.data(), static_cast<std::streamsize>(z.size()));
    for (int e = 0; e < n_experts; ++e) {
      std::vector<char> b(t.plane);
      for (size_t j = 0; j < b.size(); ++j) b[j] = static_cast<char>(pat(t.L, e, t.kind, j));
      w.o.write(b.data(), static_cast<std::streamsize>(b.size()));
    }
    written = t.off + t.plane * static_cast<uint64_t>(n_experts);
  }
}

static bool plane_matches(const void *p, int L, int E, int k, size_t n) {
  const uint8_t *b = static_cast<const uint8_t *>(p);
  for (size_t j = 0; j < n; ++j) if (b[j] != pat(L, E, k, j)) return false;
  return true;
}

// ------------------------------------------------------------------------------------------ I/O tests
static void test_store_reader(const std::string &dir) {
  const std::string path = dir + "/store.gguf";
  std::map<int, uint64_t> plane = {{3, 8192}, {4, 12288}};
  std::vector<ExpertKey> ex = {{3, 0}, {3, 5}, {3, 7}, {4, 1}, {4, 2}};
  write_store(path, ex, plane);
  OffloadStore st; std::string err;
  CHECK(st.open(path, &err));
  CHECK(st.size() == ex.size() && st.alignment() == 4096);
  CHECK(st.contains(3, 5) && !st.contains(3, 6) && !st.contains(5, 0));
  CHECK(st.keys() == ex);
  uint64_t pb[3];
  CHECK(st.plane_bytes(4, 2, pb) && pb[0] == 12288 && pb[1] == 12288 && pb[2] == 12288);
  for (const ExpertKey &k : ex) { // aligned destinations: zero-copy path (if O_DIRECT is available here)
    const uint64_t n = plane.at(k.layer);
    void *d[3];
    for (int i = 0; i < 3; ++i) CHECK(::posix_memalign(&d[i], 4096, n) == 0);
    CHECK(st.read_expert(k.layer, k.expert, d) == ReadStatus::Ok);
    for (int i = 0; i < 3; ++i) { CHECK(plane_matches(d[i], k.layer, k.expert, i, n)); std::free(d[i]); }
  }
  if (st.direct()) CHECK(st.direct_reads() == ex.size());
  std::fprintf(stderr, "  store reader: O_DIRECT %s (direct reads %llu, buffered %llu)\n", st.direct() ? "ON" : "unavailable here",
               static_cast<unsigned long long>(st.direct_reads()), static_cast<unsigned long long>(st.buffered_reads()));
  { // misaligned destinations take the buffered fallback and still return the right bytes
    std::vector<uint8_t> raw(3 * 8192 + 64);
    void *d[3] = {raw.data() + 1, raw.data() + 8192 + 3, raw.data() + 2 * 8192 + 5};
    const uint64_t buffered_before = st.buffered_reads();
    CHECK(st.read_expert(3, 7, d) == ReadStatus::Ok);
    for (int i = 0; i < 3; ++i) CHECK(plane_matches(d[i], 3, 7, i, 8192));
    CHECK(st.buffered_reads() == buffered_before + 1);
  }
  { void *d[3] = {nullptr, nullptr, nullptr}; CHECK(st.read_expert(3, 6, d) == ReadStatus::NotFound); }
  { // not a store / missing file / a store with a non-tight expert (read via the buffered path, still correct)
    OffloadStore bad; std::string e2;
    CHECK(!bad.open(dir + "/missing.gguf", &e2) && !e2.empty());
    const std::string mp = dir + "/model_for_store_test.gguf";
    write_model(mp, {3}, 4, {{3, 8192}});
    CHECK(!bad.open(mp, &e2));
    const std::string gp = dir + "/gap_store.gguf";
    write_store(gp, {{3, 1}}, {{3, 8192}}, true);
    OffloadStore gap;
    CHECK(gap.open(gp, &e2));
    void *d[3];
    for (int i = 0; i < 3; ++i) CHECK(::posix_memalign(&d[i], 4096, 8192) == 0);
    CHECK(gap.read_expert(3, 1, d) == ReadStatus::Ok && gap.direct_reads() == 0);
    for (int i = 0; i < 3; ++i) { CHECK(plane_matches(d[i], 3, 1, i, 8192)); std::free(d[i]); }
  }
}

static void test_model_reader(const std::string &dir) {
  const std::string path = dir + "/model.gguf";
  std::map<int, uint64_t> plane = {{3, 8192}, {5, 4096}};
  write_model(path, {3, 5}, 6, plane);
  ModelReader m; std::string err;
  CHECK(m.open({path}, &err));
  CHECK(m.contains(3, 5) && m.contains(5, 0) && !m.contains(3, 6) && !m.contains(4, 0) && !m.contains(3, -1));
  uint64_t pb[3];
  CHECK(m.plane_bytes(5, 2, pb) && pb[0] == 4096 && pb[2] == 4096);
  for (int L : {3, 5}) for (int E : {0, 3, 5}) {
    const uint64_t n = plane.at(L);
    std::vector<uint8_t> a(n), b(n), c(n);
    void *d[3] = {a.data(), b.data(), c.data()};
    CHECK(m.read_expert(L, E, d) == ReadStatus::Ok);
    CHECK(plane_matches(a.data(), L, E, 0, n) && plane_matches(b.data(), L, E, 1, n) && plane_matches(c.data(), L, E, 2, n));
  }
  ModelReader none; CHECK(!none.open({dir + "/missing.gguf"}));
}

static void test_arena() {
  VirtualArena a(3, 2, 8, {{8192, 8192, 8192}, {5000, 4096, 4096}}, true);
  CHECK(a.ok());
  CHECK(a.stride(4, 0) == 8192 && a.plane_bytes(4, 0) == 5000 && a.stride(3, 1) == 8192);
  CHECK(reinterpret_cast<uintptr_t>(a.ptr(3, 2, 0)) % 4096 == 0);
  CHECK(static_cast<char *>(a.ptr(3, 3, 0)) - static_cast<char *>(a.ptr(3, 2, 0)) == 8192); // slot index == expert id
  CHECK(a.resident_pages(3, 2) == 0); // nothing populated yet: NORESERVE costs no memory
  std::memset(a.ptr(3, 2, 0), 0xAB, 8192); std::memset(a.ptr(3, 2, 1), 0xCD, 8192); std::memset(a.ptr(3, 2, 2), 0xEF, 8192);
  std::memset(a.ptr(3, 3, 0), 0x11, 8192);
  CHECK(a.resident_pages(3, 2) == 6 && a.resident_pages(3, 3) == 2);
  a.evict(3, 2);
  CHECK(a.resident_pages(3, 2) == 0 && a.resident_pages(3, 3) == 2); // the neighbour is untouched
  CHECK(static_cast<uint8_t *>(a.ptr(3, 3, 0))[100] == 0x11);
  CHECK(static_cast<uint8_t *>(a.ptr(3, 2, 0))[100] == 0); // an evicted slot reads as zeros until repopulated
  a.evict(3, 2); // idempotent
  CHECK(a.resident_pages(3, 2) == 0);
}

struct Zipf {
  std::mt19937 rng; std::vector<ExpertKey> keys; std::geometric_distribution<int> g{0.03};
  Zipf(unsigned seed, std::vector<ExpertKey> k) : rng(seed), keys(std::move(k)) {}
  ExpertKey next() { return keys[static_cast<size_t>(std::min<int>(g(rng), static_cast<int>(keys.size()) - 1))]; }
};

static void test_residency_end_to_end(const std::string &dir) {
  const int first = 3, nl = 3, ne = 32; const uint64_t P = 8192;
  std::map<int, uint64_t> plane = {{3, P}, {4, P}, {5, P}};
  std::vector<ExpertKey> universe = all_keys(first, nl, ne), in_store, in_model_only, nowhere;
  std::mt19937 pr(777);
  for (const ExpertKey &k : universe) {
    int r = static_cast<int>(pr() % 10);
    if (k.layer == 5 && k.expert == 31) nowhere.push_back(k);   // no source has it: loading must fail loudly
    else if (r < 6) in_store.push_back(k);
    else in_model_only.push_back(k);
  }
  write_store(dir + "/e2e_store.gguf", in_store, plane);
  // the model holds every expert except the one that is "nowhere": give it a model without layer 5's last expert by
  // pointing the model at only 31 experts of layer 5 is not possible (tensor shape), so instead the model covers
  // layers 3 and 4 fully and layer 5 for all experts; the "nowhere" key is excluded by hiding it behind a filter source.
  write_model(dir + "/e2e_model.gguf", {3, 4, 5}, ne, plane);
  OffloadStore store; ModelReader model; std::string err;
  CHECK(store.open(dir + "/e2e_store.gguf", &err) && model.open({dir + "/e2e_model.gguf"}, &err));
  struct Filter final : ExpertSource { // hides one key from the model, to inject a load failure
    ModelReader &m; ExpertKey hide;
    Filter(ModelReader &mm, ExpertKey h) : m(mm), hide(h) {}
    bool contains(int l, int e) const override { return !(l == hide.layer && e == hide.expert) && m.contains(l, e); }
    bool plane_bytes(int l, int e, uint64_t o[3]) const override { return contains(l, e) && m.plane_bytes(l, e, o); }
    ReadStatus read_expert(int l, int e, void *const d[3]) override { return m.read_expert(l, e, d); }
  } filtered(model, nowhere.front());

  ExpertCache cache({first, nl, ne, 24, ExpertCache::Policy::Lru, 0.25});
  VirtualArena arena(first, nl, ne, {{P, P, P}, {P, P, P}, {P, P, P}}, true);
  ExpertResidency res(cache, arena, {&store, &filtered});
  RefLru ref{24, {}, {}};
  // seed the hottest 24 and load them (as at startup)
  std::vector<ExpertKey> hot; for (const ExpertKey &k : universe) if (!(k == nowhere.front())) hot.push_back(k);
  std::shuffle(hot.begin(), hot.end(), pr); hot.resize(24);
  cache.seed(hot); ref.seed(hot, 0.25);
  CHECK(res.populate_resident() == 0);
  Zipf z(99, universe);
  uint64_t misses = 0, ref_misses = 0, loads = 0, errors = 0;
  for (int step = 0; step < 6000; ++step) {
    const int layer = first + (step % nl);
    std::vector<int32_t> ids; ExpertKey probe[8];
    for (int i = 0; i < 8; ++i) { ExpertKey k; do { k = z.next(); } while (k.layer != layer); ids.push_back(k.expert); probe[i] = k; }
    ids.push_back(-1); // a skipped route
    auto e = res.ensure(layer, ids.data(), static_cast<int>(ids.size()));
    misses += static_cast<uint64_t>(e.misses); loads += static_cast<uint64_t>(e.loads); errors += static_cast<uint64_t>(e.errors);
    bool had_error_key = false;
    for (int i = 0; i < 8; ++i) {
      if (probe[i] == nowhere.front()) { had_error_key = true; continue; }
      auto rr = ref.access(probe[i]); ref_misses += rr.hit ? 0 : 1;
    }
    if (!had_error_key) { // the reference never sees the unloadable key, so only compare when it was not routed
      CHECK(cache.all_resident(layer, ids.data(), static_cast<int>(ids.size())));
      for (int i = 0; i < 8; ++i) {
        CHECK(arena.resident_pages(probe[i].layer, probe[i].expert) > 0);
        for (int k = 0; k < 3; ++k) CHECK(plane_matches(arena.ptr(probe[i].layer, probe[i].expert, k), probe[i].layer, probe[i].expert, k, P));
      }
    }
    if (step % 500 == 0) { // bitmap <=> cache <=> arena residency, for every key
      for (const ExpertKey &k : universe) {
        const bool r = cache.resident(k);
        const uint64_t *b = cache.layer_bitmap(k.layer);
        CHECK(((b[k.expert >> 6] >> (k.expert & 63)) & 1ull) == (r ? 1ull : 0ull));
        CHECK((arena.resident_pages(k.layer, k.expert) > 0) == r);
      }
      CHECK(cache.size() <= 24);
    }
  }
  CHECK(errors > 0 && !cache.resident(nowhere.front()));       // the unloadable expert was reported and never cached
  CHECK(loads > 0 && misses == loads + errors);                 // every miss was either loaded or reported
  std::fprintf(stderr, "  residency e2e: %llu misses, %llu loads, %llu reported errors; store direct reads %llu, buffered %llu\n",
               static_cast<unsigned long long>(misses), static_cast<unsigned long long>(loads), static_cast<unsigned long long>(errors),
               static_cast<unsigned long long>(store.direct_reads()), static_cast<unsigned long long>(store.buffered_reads()));
  (void)ref_misses;
  // a Static cache loads nothing and reports every miss as not cached
  ExpertCache st({first, nl, ne, 4, ExpertCache::Policy::Static, 0.0});
  VirtualArena arena2(first, nl, ne, {{P, P, P}, {P, P, P}, {P, P, P}}, true);
  ExpertResidency res2(st, arena2, {&store});
  st.seed({{3, 0}});
  const int32_t ids2[] = {0, 1};
  auto e2 = res2.ensure(3, ids2, 2);
  CHECK(e2.misses == 1 && e2.not_cached == 1 && e2.loads == 0 && e2.errors == 0 && arena2.resident_pages(3, 1) == 0);

  // Prefill path: loads non-resident experts without touching the cache, and drops them at the next call.
  ExpertCache lru({first, nl, ne, 4, ExpertCache::Policy::Lru, 0.0});
  VirtualArena arena3(first, nl, ne, {{P, P, P}, {P, P, P}, {P, P, P}}, true);
  ExpertResidency res3(lru, arena3, {&store, &model});
  lru.seed({{3, 0}});
  res3.load(3, 0);
  const int32_t ids3[] = {0, 1, 1, 2, -1};
  const auto t1 = res3.ensure_transient(3, ids3, 5);
  CHECK(t1.misses == 2 && t1.loads == 2 && t1.errors == 0 && lru.size() == 1);
  CHECK(arena3.resident_pages(3, 1) > 0 && arena3.resident_pages(3, 2) > 0 && arena3.resident_pages(3, 0) > 0);
  const int32_t ids4[] = {4};
  const auto t2 = res3.ensure_transient(4, ids4, 1);
  CHECK(t2.loads == 1 && arena3.resident_pages(3, 1) == 0 && arena3.resident_pages(3, 2) == 0 && lru.size() == 1);
}

// ------------------------------------------------------------------------------------------ acceptance: trace replay
struct Trace { std::vector<std::vector<std::vector<std::pair<int, std::vector<int>>>>> prompts; }; // prompt -> decode tokens -> (layer, ids)

static bool load_trace(const std::string &path, int lo, int hi, std::vector<std::vector<std::vector<std::pair<int, std::vector<int>>>>> &prompts) {
  std::ifstream in(path);
  if (!in) return false;
  using Sweep = std::vector<std::pair<int, std::vector<int>>>;
  std::vector<Sweep> cur; Sweep sweep; int last = -1; std::string line;
  auto flush = [&]() { if (!sweep.empty()) { cur.push_back(std::move(sweep)); sweep.clear(); } };
  while (std::getline(in, line)) {
    size_t t = line.find("[TRACE l=");
    if (t != std::string::npos) {
      int L = std::atoi(line.c_str() + t + 9);
      size_t p = line.find("router picks:");
      std::vector<int> ids;
      for (size_t i = p; (i = line.find(" e", i)) != std::string::npos && i + 2 < line.size(); i += 2)
        if (std::isdigit(static_cast<unsigned char>(line[i + 2]))) ids.push_back(std::atoi(line.c_str() + i + 2));
      if (L <= last) flush();
      sweep.emplace_back(L, ids); last = L;
    } else if (line.find("[REQUEST] ") != std::string::npos && line.find(" tokens in ") != std::string::npos) {
      flush(); last = -1;
      if (!cur.empty()) {
        std::vector<std::vector<std::pair<int, std::vector<int>>>> toks;
        for (size_t i = 1; i < cur.size(); ++i) { // sweep 0 = last prefill token
          Sweep f; for (auto &x : cur[i]) if (x.first >= lo && x.first <= hi) f.push_back(x);
          toks.push_back(std::move(f));
        }
        prompts.push_back(std::move(toks)); cur.clear();
      }
    }
  }
  return !prompts.empty();
}

struct Oracle { size_t C = 0; double mean = 0, first20 = 0, after20 = 0; };
static std::map<std::string, Oracle> load_oracle(const std::string &path) {
  std::map<std::string, Oracle> m; std::ifstream in(path); std::string line, key;
  while (std::getline(in, line)) {
    size_t a = line.find('"');
    if (a == std::string::npos) continue;
    size_t b = line.find('"', a + 1);
    std::string name = line.substr(a + 1, b - a - 1);
    if (name.find(':') != std::string::npos) { key = name; m[key] = {}; continue; }
    double v = std::strtod(line.c_str() + line.find(':', b) + 1, nullptr);
    if (key.empty()) continue;
    if (name == "C") m[key].C = static_cast<size_t>(v);
    else if (name == "misses_per_token") m[key].mean = v;
    else if (name == "first20") m[key].first20 = v;
    else if (name == "after20") m[key].after20 = v;
  }
  return m;
}

static void test_trace_replay_matches_simulator() {
  const std::string trace = "docs/ssd-offload/data/glm53flash_survey5_router_trace.txt";
  const std::string usage = "docs/ssd-offload/data/glm53flash_usage_old_survey.txt";
  const std::string oracle = "docs/ssd-offload/data/cache_sim_results_20261008.json";
  const int lo = 11, hi = 44, ne = 288;
  std::vector<std::vector<std::vector<std::pair<int, std::vector<int>>>>> prompts;
  if (!load_trace(trace, lo, hi, prompts) || !std::ifstream(usage) || !std::ifstream(oracle)) {
    std::fprintf(stderr, "  acceptance replay SKIPPED: run from the repo root (needs %s, %s, %s)\n", trace.c_str(), usage.c_str(), oracle.c_str());
    return;
  }
  auto want = load_oracle(oracle);
  CHECK(prompts.size() == 5 && prompts[0].size() == 159 && prompts[4].size() == 80);
  auto rank = rank_hottest_first(usage, lo, hi - lo + 1, ne);
  CHECK(rank.size() == static_cast<size_t>((hi - lo + 1) * ne));
  int compared = 0;
  for (double B : {32.0, 40.0, 48.0, 56.0}) {
    const size_t C = static_cast<size_t>(B * 1073741824.0 / 6.49e6);
    struct Case { const char *policy; bool session; };
    const Case cases[] = {{"static", true}, {"LRU-empty", false}, {"LRU-empty", true}, {"static+LRU", false}, {"static+LRU", true}, {"pinned+LRU", false}, {"pinned+LRU", true}};
    for (const Case &cs : cases) {
      const std::string pol = cs.policy;
      ExpertCache::Config cfg{lo, hi - lo + 1, ne, C, pol == "static" ? ExpertCache::Policy::Static : ExpertCache::Policy::Lru, pol == "pinned+LRU" ? 0.5 : 0.0};
      auto fresh = [&]() { ExpertCache c(cfg); if (pol != "LRU-empty") c.seed(std::vector<ExpertKey>(rank.begin(), rank.begin() + static_cast<long>(std::min(C, rank.size())))); return c; };
      ExpertCache cache = fresh();
      uint64_t all = 0, ntok = 0, f20 = 0, nf20 = 0, a20 = 0, na20 = 0;
      for (auto &p : prompts) {
        if (!cs.session) cache = fresh();
        size_t ti = 0;
        for (auto &tok : p) {
          uint64_t miss = 0;
          for (auto &[L, ids] : tok) for (int e : ids) miss += cache.access({L, e}).hit ? 0 : 1;
          all += miss; ++ntok;
          if (ti < 20) { f20 += miss; ++nf20; } else { a20 += miss; ++na20; }
          ++ti;
        }
      }
      char key[96]; std::snprintf(key, sizeof key, "%.1f:%s:%s", B, cs.policy, pol == "static" ? "session" : (cs.session ? "session" : "per-prompt"));
      auto it = want.find(key);
      CHECK(it != want.end());
      if (it == want.end()) continue;
      const double mean = static_cast<double>(all) / static_cast<double>(ntok), m1 = static_cast<double>(f20) / static_cast<double>(nf20), m2 = static_cast<double>(a20) / static_cast<double>(na20);
      const bool ok = it->second.C == C && std::fabs(mean - it->second.mean) < 1e-9 && std::fabs(m1 - it->second.first20) < 1e-9 && std::fabs(m2 - it->second.after20) < 1e-9;
      CHECK(ok);
      if (!ok) std::fprintf(stderr, "  MISMATCH %s: C++ mean %.9f first20 %.9f after20 %.9f vs simulator %.9f %.9f %.9f (C %zu vs %zu)\n", key, mean, m1, m2, it->second.mean, it->second.first20, it->second.after20, C, it->second.C);
      ++compared;
    }
  }
  std::fprintf(stderr, "  acceptance replay: %d of %zu simulator results reproduced exactly (mean, first-20, after-20 misses per token)\n", compared, want.size());
  CHECK(compared == 28 && want.size() == 28);
}

// ------------------------------------------------------------------------------------------ optional: real store + model
// MOESTREAM_TEST_STORE=<store.gguf> MOESTREAM_TEST_MODEL=<shard1.gguf>[,<shard2.gguf>...] [MOESTREAM_TEST_SAMPLE=N]
static void test_real_store() {
  const char *sp = std::getenv("MOESTREAM_TEST_STORE"), *mp = std::getenv("MOESTREAM_TEST_MODEL");
  if (!sp || !mp) { std::fprintf(stderr, "  real store/model test SKIPPED (set MOESTREAM_TEST_STORE and MOESTREAM_TEST_MODEL)\n"); return; }
  std::vector<std::string> shards; { std::stringstream ss(mp); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) shards.push_back(t); }
  OffloadStore st; ModelReader model; std::string err;
  CHECK(st.open(sp, &err));
  CHECK(model.open(shards, &err));
  if (g_failed) { std::fprintf(stderr, "  real open failed: %s\n", err.c_str()); return; }
  auto keys = st.keys();
  const char *sm = std::getenv("MOESTREAM_TEST_SAMPLE");
  const size_t stride = sm && std::atoi(sm) > 0 ? std::max<size_t>(1, keys.size() / static_cast<size_t>(std::atoi(sm))) : 1;
  size_t n = 0, bad = 0; double read_ms = 0; uint64_t bytes = 0;
  for (size_t i = 0; i < keys.size(); i += stride) {
    const ExpertKey k = keys[i];
    uint64_t sb[3], mb[3];
    if (!st.plane_bytes(k.layer, k.expert, sb) || !model.plane_bytes(k.layer, k.expert, mb) || std::memcmp(sb, mb, sizeof sb) != 0) { ++bad; continue; }
    void *a[3]; std::vector<uint8_t> b[3]; void *bp[3];
    for (int j = 0; j < 3; ++j) { if (::posix_memalign(&a[j], 4096, static_cast<size_t>(sb[j]))) std::abort(); b[j].resize(static_cast<size_t>(sb[j])); bp[j] = b[j].data(); }
    const auto t0 = std::chrono::steady_clock::now();
    const bool r1 = st.read_expert(k.layer, k.expert, a) == ReadStatus::Ok;
    read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const bool r2 = model.read_expert(k.layer, k.expert, bp) == ReadStatus::Ok;
    bool same = r1 && r2;
    for (int j = 0; j < 3 && same; ++j) same = std::memcmp(a[j], bp[j], static_cast<size_t>(sb[j])) == 0;
    if (!same) ++bad;
    bytes += sb[0] + sb[1] + sb[2];
    for (int j = 0; j < 3; ++j) std::free(a[j]);
    ++n;
  }
  CHECK(bad == 0 && n > 0);
  std::fprintf(stderr, "  real store: %zu experts compared with the main model, %zu mismatches; store reads via %s: %.2f ms/expert avg, %.2f GB/s (%llu direct, %llu buffered)\n",
               n, bad, st.direct() ? "O_DIRECT preadv" : "buffered", n ? read_ms / static_cast<double>(n) : 0.0, read_ms > 0 ? static_cast<double>(bytes) / 1e9 / (read_ms / 1e3) : 0.0,
               static_cast<unsigned long long>(st.direct_reads()), static_cast<unsigned long long>(st.buffered_reads()));
}

int main() {
  const std::string dir = test_dir();
  std::fprintf(stderr, "[expert_dram_cache] temp dir %s\n", dir.c_str());
  test_random_equivalence();   std::fprintf(stderr, "  random equivalence with the reference LRU done\n");
  test_edge_cases();
  test_bitmap_all_resident();
  test_ranking(dir);
  test_store_reader(dir);
  test_model_reader(dir);
  test_arena();
  test_residency_end_to_end(dir);
  test_trace_replay_matches_simulator();
  test_real_store();
  std::string rm = "rm -rf '" + dir + "'";
  if (std::system(rm.c_str()) != 0) std::fprintf(stderr, "warning: could not remove %s\n", dir.c_str());
  std::fprintf(stderr, "[expert_dram_cache] %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
