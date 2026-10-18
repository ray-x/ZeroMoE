#pragma once

#include <cerrno>
#include <cstring>
#include <expected>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

inline std::vector<std::string> exllamav3_backend_arguments(
    const std::string &config_path, const std::string &python,
    const std::string &runner, const std::vector<std::string> &overrides) {
  if (python.empty() || runner.empty())
    throw std::invalid_argument("exllamav3_python and exllamav3_runner_path are required");
  const auto config = std::filesystem::absolute(config_path);
  const auto resolve = [&](const std::string &value) {
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : config.parent_path() / path)
        .lexically_normal().string();
  };
  std::vector<std::string> arguments{
      python.find('/') == std::string::npos ? python : resolve(python),
      resolve(runner), "--config", config.string()};
  arguments.insert(arguments.end(), overrides.begin(), overrides.end());
  return arguments;
}

inline std::expected<void, std::string> exec_exllamav3_backend(
    const std::string &config_path, const std::string &python,
    const std::string &runner, const std::vector<std::string> &overrides) {
  try {
    auto arguments = exllamav3_backend_arguments(config_path, python, runner, overrides);
    std::vector<char *> pointers;
    for (auto &argument : arguments)
      pointers.push_back(argument.data());
    pointers.push_back(nullptr);
    execvp(pointers.front(), pointers.data());
    return std::unexpected("Cannot execute ExLlamaV3 backend: " + std::string(std::strerror(errno)));
  } catch (const std::exception &error) {
    return std::unexpected(error.what());
  }
}
