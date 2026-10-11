#pragma once
// Optional learned correction for a next-layer expert pilot (bench/qwen_ridge_export*.py, bench/glm_ridge_export.py).
// score = pilot score + (x - mu) R + c, one record per target layer. File: int32 magic 'RDG1' (0x31474452), D, E, n; per record
// int32 target layer, f32 mu[D], f16 R^T [E][D], f32 c[E]. Loaded once per environment variable; a missing or invalid file leaves the
// plain pilot.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct PilotRidgeTable {
  struct Record { std::vector<float> mu, c; std::vector<uint16_t> r; };
  std::map<int, Record> by_layer;
  int d = 0, e = 0;
  const Record *find(int layer) const {
    const auto it = by_layer.find(layer);
    return it == by_layer.end() ? nullptr : &it->second;
  }
  static PilotRidgeTable load(const char *env_name, const char *tag) {
    PilotRidgeTable t;
    const char *path = std::getenv(env_name);
    if (!path || !path[0]) return t;
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[%s] cannot open %s\n", tag, path); return t; }
    int32_t hdr[4];
    bool ok = std::fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x31474452 && hdr[1] > 0 && hdr[2] > 0;
    if (ok) {
      t.d = hdr[1]; t.e = hdr[2];
      for (int i = 0; ok && i < hdr[3]; ++i) {
        int32_t layer;
        Record rec;
        rec.mu.resize(static_cast<size_t>(t.d)); rec.c.resize(static_cast<size_t>(t.e));
        rec.r.resize(static_cast<size_t>(t.d) * static_cast<size_t>(t.e));
        ok = std::fread(&layer, 4, 1, f) == 1 &&
             std::fread(rec.mu.data(), 4, rec.mu.size(), f) == rec.mu.size() &&
             std::fread(rec.r.data(), 2, rec.r.size(), f) == rec.r.size() &&
             std::fread(rec.c.data(), 4, rec.c.size(), f) == rec.c.size();
        if (ok) t.by_layer.emplace(layer, std::move(rec));
      }
    }
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "[%s] invalid file %s\n", tag, path); t.by_layer.clear(); }
    else std::fprintf(stderr, "[%s] loaded %zu layer corrections from %s\n", tag, t.by_layer.size(), path);
    return t;
  }
};
