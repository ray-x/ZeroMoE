#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtp_test {

struct Row {
  int token = 0, position = 0;
  std::vector<float> embedding, hidden;
};

struct Fixture {
  int D = 0, W = 0, V = 0;
  uint64_t fingerprint = 14695981039346656037ULL;
  std::vector<Row> rows;

  explicit Fixture(const char *path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open MTP fixture");
    char byte;
    while (input.get(byte)) {
      fingerprint ^= static_cast<unsigned char>(byte);
      fingerprint *= 1099511628211ULL;
    }
    if (!input.eof()) throw std::runtime_error("cannot read MTP fixture");
    input.clear();
    input.seekg(0);
    std::string magic;
    int count = 0;
    if (!(input >> magic >> D >> W >> V >> count) ||
        magic != "ZEROMOE_MTP_FIXTURE_V1" ||
        D <= 0 || W < D || D > (1 << 20) || W > (1 << 20) ||
        V <= 0 || V > (1 << 24) || count <= 0 || count > 4096 ||
        uint64_t(count) * (uint64_t(D) + W) > (1 << 26))
      throw std::runtime_error("invalid MTP fixture header or dimensions");
    rows.resize(count);
    for (int i = 0; i < count; ++i) {
      auto &row = rows[i];
      if (!(input >> row.token >> row.position) ||
          row.token < 0 || row.token >= V || row.position != i)
        throw std::runtime_error("invalid fixture token or nonsequential position");
      row.embedding.resize(D);
      row.hidden.resize(W);
      for (auto *values : {&row.embedding, &row.hidden})
        for (float &value : *values)
          if (!(input >> value) || !std::isfinite(value))
            throw std::runtime_error("missing/non-finite fixture input");
    }
    input >> std::ws;
    if (!input.eof()) throw std::runtime_error("unexpected trailing fixture data");
  }

  void dump_header(std::ostream &out) const {
    out << "ZEROMOE_MTP_DUMP_V1 " << fingerprint << ' ' << D << ' '
        << W << ' ' << V << ' ' << rows.size() << '\n';
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
  }
};

template <typename T>
void dump(std::ostream &out, int phase, int row, const std::string &stage,
          const std::vector<T> &values) {
  out << phase << ' ' << row << ' ' << stage << ' ' << values.size();
  for (auto value : values) out << ' ' << value;
  out << '\n';
  if (!out) throw std::runtime_error("MTP dump write failed");
}

} // namespace mtp_test
