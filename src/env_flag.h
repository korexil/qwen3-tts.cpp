#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace qwen3_tts {

// One rule for every QWEN3_TTS_* switch: "1", "true", "on" or "yes" (any case) turn it
// on; unset, empty, "0", "false", "off", "no" and anything else leave it off.
inline bool env_flag_on(const char * key) {
    const char * value = std::getenv(key);
    if (!value || !*value) {
        return false;
    }
    char lowered[8] = {0};
    const size_t length = std::strlen(value);
    if (length >= sizeof(lowered)) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        lowered[i] = (char) std::tolower((unsigned char) value[i]);
    }
    return !std::strcmp(lowered, "1") || !std::strcmp(lowered, "true") ||
           !std::strcmp(lowered, "on") || !std::strcmp(lowered, "yes");
}

} // namespace qwen3_tts
