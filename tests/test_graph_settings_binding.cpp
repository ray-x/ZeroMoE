#include "graph_settings_global.hpp"
#include <cstdio>

using namespace zeromoe::graphsettings;

int main() {
  try {
    configure_settings({{"device_expert_layers", "49"}});
    bool invalid_rejected = false;
    try { settings_for(Family::Qwen); }
    catch (const std::invalid_argument &) { invalid_rejected = true; }
    if (!invalid_rejected) throw std::runtime_error("invalid configuration was silently accepted");
    configure_settings({{"device_expert_layers", "4"}, {"device_expert_trailing", "true"}});
    const auto &settings = settings_for(Family::Qwen);
    if (settings.tun.device_expert_layers.value != 4 ||
        settings.tun.device_expert_trailing.value != 1 ||
        &settings != &settings_for(Family::Qwen))
      throw std::runtime_error("configured graph settings did not remain stable");
    if (settings_for(Family::Other).tun.phase_cache_mib.value != 0)
      throw std::runtime_error("Qwen settings leaked into shared phase cache");
    bool rejected = false;
    try { configure_settings({}); }
    catch (const std::logic_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("settings changed after graph initialization");
    std::puts("graph settings binding: OK");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
