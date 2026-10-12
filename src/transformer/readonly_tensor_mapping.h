#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qwen3_tts {

// Owns a read-only tensor view. GGML's from_ptr buffer owns only metadata;
// free that buffer before destroying this mapping. No heap copy or prefault.
class ReadonlyTensorMapping {
public:
    ReadonlyTensorMapping(const ReadonlyTensorMapping &) = delete;
    ReadonlyTensorMapping & operator=(const ReadonlyTensorMapping &) = delete;

    ~ReadonlyTensorMapping() {
#ifdef _WIN32
        if (view_) UnmapViewOfFile(view_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (view_) munmap(view_, view_bytes_);
        if (file_ >= 0) close(file_);
#endif
    }

    void * data() const { return static_cast<char *>(view_) + delta_; }
    size_t size() const { return tensor_bytes_; }

    static std::unique_ptr<ReadonlyTensorMapping> open(
            const std::string & path, uint64_t offset, size_t bytes, std::string & error) {
        auto result = std::unique_ptr<ReadonlyTensorMapping>(new ReadonlyTensorMapping());
        if (!bytes) { error = "Cannot map an empty tensor"; return nullptr; }
        uint64_t file_bytes = 0, granularity = 0;
#ifdef _WIN32
        if (path.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            error = "UTF-8 model path too long"; return nullptr;
        }
        const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                              path.data(), static_cast<int>(path.size()), nullptr, 0);
        if (!length) { error = "Invalid UTF-8 model path"; return nullptr; }
        std::wstring wide(static_cast<size_t>(length), L'\0');
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(),
                              static_cast<int>(path.size()), &wide[0], length) != length) {
            error = "Failed to convert model path"; return nullptr;
        }
        result->file_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (result->file_ == INVALID_HANDLE_VALUE) {
            error = "Cannot open read-only tensor file: " + std::to_string(GetLastError()); return nullptr;
        }
        LARGE_INTEGER size;
        if (!GetFileSizeEx(result->file_, &size) || size.QuadPart < 0) {
            error = "Cannot determine tensor file size"; return nullptr;
        }
        file_bytes = static_cast<uint64_t>(size.QuadPart);
        SYSTEM_INFO info; GetSystemInfo(&info);
        granularity = info.dwAllocationGranularity;
#else
        result->file_ = ::open(path.c_str(), O_RDONLY);
        struct stat info;
        if (result->file_ < 0 || fstat(result->file_, &info) || info.st_size < 0) {
            error = "Cannot open/stat read-only tensor file"; return nullptr;
        }
        file_bytes = static_cast<uint64_t>(info.st_size);
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) { error = "Invalid mmap page size"; return nullptr; }
        granularity = static_cast<uint64_t>(page_size);
#endif
        if (!granularity || offset > file_bytes || bytes > file_bytes - offset) {
            error = "Tensor mapping exceeds model file bounds"; return nullptr;
        }
        const uint64_t base_offset = offset - offset % granularity;
        const uint64_t delta = offset - base_offset;
        if (delta > std::numeric_limits<size_t>::max() - bytes) {
            error = "Tensor mapping size overflow"; return nullptr;
        }
        result->delta_ = static_cast<size_t>(delta);
        result->tensor_bytes_ = bytes;
        result->view_bytes_ = bytes + result->delta_;
#ifdef _WIN32
        result->mapping_ = CreateFileMappingW(result->file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!result->mapping_) { error = "CreateFileMapping failed"; return nullptr; }
        result->view_ = MapViewOfFile(result->mapping_, FILE_MAP_READ,
            static_cast<DWORD>(base_offset >> 32), static_cast<DWORD>(base_offset), result->view_bytes_);
        if (!result->view_) { error = "MapViewOfFile failed"; return nullptr; }
#else
        if (base_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            error = "Tensor offset exceeds off_t"; return nullptr;
        }
        void * view = mmap(nullptr, result->view_bytes_, PROT_READ, MAP_PRIVATE,
                          result->file_, static_cast<off_t>(base_offset));
        if (view == MAP_FAILED) { error = "mmap failed"; return nullptr; }
        result->view_ = view;
#endif
        return result;
    }

private:
    ReadonlyTensorMapping() = default;
    void * view_ = nullptr;
    size_t view_bytes_ = 0, delta_ = 0, tensor_bytes_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_ = -1;
#endif
};

} // namespace qwen3_tts
