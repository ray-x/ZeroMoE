#pragma once

#include "graph_settings.hpp"
#include <type_traits>

namespace zeromoe::graphsettings {

// Both existing TOML readers expose the same table/value interface.
template<class Document>
bool collect_config(const Document &doc, ConfigValues &values, std::string &error) {
  values.clear();
  for (const auto &table : doc.gpu_tables)
    for (const auto &[key, value] : table.values) {
      (void)value;
      if (key.starts_with("graph_"))
        error += key + " must be a root setting before [[gpus]]\n";
    }
  for (const auto &[key, value] : doc.root.values) {
    if (!key.starts_with("graph_")) continue;
    const auto name = key.substr(6);
    bool known = false;
    each_field([&](const auto &f) {
      if (name != f.name) return;
      known = true;
      using Value = std::remove_cvref_t<decltype(value)>;
      bool type_ok = false;
      if constexpr (std::is_same_v<std::remove_cvref_t<decltype(f.dflt)>, std::string>)
        type_ok = value.type == Value::String;
      else if (auto_flag(f.name, f.parse))
        type_ok = value.type == Value::Bool || (value.type == Value::String && value.str == "auto");
      else if (boolean(f.parse))
        type_ok = value.type == Value::Bool;
      else
        type_ok = value.type == Value::Number ||
                  (f.parse == Parse::RealOrAuto && value.type == Value::String && value.str == "auto");
      if (!type_ok) { error += key + ": incorrect TOML type\n"; return; }
      try {
        Resolved<std::remove_cvref_t<decltype(f.dflt)>> slot;
        detail::canonical(f, value.str, slot);
        values[name] = value.str;
      } catch (const std::invalid_argument &e) {
        error += key + ": " + e.what() + "\n";
      }
    });
    if (!known) error += "unknown setting " + key + "\n";
  }
  return error.empty();
}
}
