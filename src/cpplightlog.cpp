#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <optional>

namespace nb = nanobind;
namespace fs = std::filesystem;

/**
 * @brief A C++ logger class that provides core logging functionality for `LightLog`
 *
 * This class implements a logging workflow that can output to both console and file.
 * It supports rank-based logging for distributed systems and various log levels.
 */
class CppLogger : public std::streambuf
{
public:
    /**
     * @brief Construct a new CppLogger object
     *
     * @param name The name of the logger
     * @param file_path Path to the log file (optional)
     * @param mode File opening mode (default: "a" for append)
     * @param level Logging level (default: 20 for INFO)
     * @param use_rank Whether to use rank-based logging (default: false)
     * @param rank Process rank for distributed logging (default: 0)
     * @param world_size Total number of processes (default: 1)
     * @param auto_detect_env Environment for auto-detecting rank and world size (default: "none")
     */
    CppLogger(const std::string &name,
              const std::string &file_path = "",
              const std::string &mode = "a",
              int level = 0,
              bool use_rank = false,
              int rank = 0,
              int world_size = 1,
              const std::string &auto_detect_env = "none",
              int log_rank = -1)
        : name_(name), file_path_(file_path), mode_(mode), level_(level),
          use_rank_(use_rank), rank_(rank), world_size_(world_size), log_rank_(log_rank)
    {
        if (use_rank_)
            std::tie(rank_, world_size_) = get_rank_and_world_size(rank, world_size, auto_detect_env);
        if (!file_path_.empty())
            open_file();
    }

    /**
     * @brief Destroy the CppLogger object
     *
     * Ensures all buffered data is flushed and files are closed.
     */
    ~CppLogger()
    {
        flush();
        close();
    }

    /**
     * @brief Flush the logger, writing any buffered data
     */
    void flush()
    {
        if (file_.is_open())
            file_.flush();
        std::cout.flush();
    }

    /**
     * @brief Close the log file if it's open
     */
    void close()
    {
        if (file_.is_open())
            file_.close();
    }

    /**
     * @brief Log a message with specified level and options
     *
     * @param msg The message to log
     * @param level The log level for this message
     * @param use_rank Whether to include rank information for this message
     * @param new_file Optional new file to log this message to
     */
    void log(const std::string &msg, int level, bool use_rank = false, const std::string &new_file = "")
    {
        if (level >= level_)
        {
            if (log_rank_ != -1 && rank_ != log_rank_)
                return; // only log on specific rank

            std::string formatted_msg = format_message(msg, level);
            formatted_msg = (use_rank || use_rank_) ? "[" + std::to_string(rank_) + "/" + std::to_string(world_size_) + "] " + formatted_msg : formatted_msg;

            std::cout << formatted_msg;

            if (!new_file.empty())
                log_to_file(formatted_msg, new_file);
            else if (file_.is_open())
                file_ << formatted_msg;
        }
    }

    /**
     * @brief Reconfigure the logger with new settings
     *
     * Updates the logger's configuration with the provided parameters. If a parameter is not provided (or is empty), the corresponding setting will not be changed.
     *
     * @param name New logger name (optional)
     * @param file_path New path to the log file (optional)
     * @param mode New file opening mode (default: "a" for append)
     * @param level New logging level (default: -1, which means no change; valid values are 0-50, where 0 is NOTSET, 10 is DEBUG, 20 is INFO, 30 is WARNING, 40 is ERROR, and 50 is CRITICAL)
     * @param use_rank New value for whether to use rank-based logging (default: false)
     * @param rank New process rank for distributed logging (default: 0)
     * @param world_size New total number of processes (default: 1)
     * @param auto_detect_env New environment for auto-detecting rank and world size (default: "none")
     * @param log_rank New log rank (default: -1, which means no change)
     */
    void reconfigure(const std::string &name,
                     const std::string &file_path,
                     const std::string &mode = "a",
                     int level = -1,
                     const bool &use_rank = false,
                     int rank = -1,
                     int world_size = -1,
                     const std::string &auto_detect_env = "",
                     const int &log_rank = -1)
    {
        name_ = !name.empty() ? name : name_;
        mode_ = !mode.empty() ? mode : mode_;
        if (!file_path.empty())
        {
            if (file_path != file_path_)
            {
                file_path_ = file_path;
                if (file_.is_open())
                    file_.close();

                open_file();
            }
        }
        level_ = level != -1 ? level : level_;
        rank_ = rank != -1 ? rank : rank_;
        world_size_ = world_size != -1 ? world_size : world_size_;
        auto_detect_env_ = !auto_detect_env.empty() ? auto_detect_env : auto_detect_env_;

        use_rank_ = use_rank;
        if (use_rank_)
            std::tie(rank_, world_size_) = get_rank_and_world_size(rank, world_size, auto_detect_env);

        log_rank_ = (log_rank != -1) ? log_rank : log_rank_;
    }

private:
    std::string name_, file_path_, mode_, auto_detect_env_;
    int level_, rank_, world_size_;
    int log_rank_ = -1;
    bool use_rank_;
    std::ofstream file_;

    /**
     * @brief Open the log file
     *
     * Creates necessary directories and opens the file stream.
     */
    void open_file()
    {
        fs::create_directories(fs::path(file_path_).parent_path());
        file_.open(file_path_, mode_ == "w" ? std::ios::trunc : std::ios::app);
        if (!file_.is_open())
            std::cerr << "Failed to open file: " << file_path_ << std::endl;
    }

    /**
     * @brief Format a log message
     *
     * @param msg The raw message
     * @param level The log level
     * @return std::string The formatted message
     */
    [[nodiscard]] std::string format_message(const std::string &msg, int level) const
    {
        if (level == 0)
        {
            return msg;
        }

        // Pre-allocate string with estimated size
        std::string result;
        result.reserve(64 + name_.size() + msg.size());

        // Get current time and milliseconds
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;

        // Format date and time
        char time_buf[20];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t_now));
        result.append(time_buf);

        // Append milliseconds
        char ms_buf[5];
        snprintf(ms_buf, sizeof(ms_buf), ",%03lu", static_cast<unsigned long>(ms));
        result.append(ms_buf);

        // Append the rest of the message
        result.append(" | ");
        result.append(name_);
        result.append(" | ");
        result.append(get_level_str(level));
        result.append(" | ");
        result.append(msg);

        return result;
    }

    // Helper function to find the exact level value
    const std::string get_level_str(int level) const
    {
        static constexpr std::pair<int, const char *> level_map[] = {
            {0, "NOTSET"}, {10, "DEBUG"}, {20, "INFO"}, {30, "WARNING"}, {40, "ERROR"}, {50, "CRITICAL"}};
        const auto size = sizeof(level_map) / sizeof(level_map[0]);

        for (const auto &pair : level_map)
        {
            if (pair.first == level)
            {
                return pair.second;
            }
        }

        return ""; // or throw an exception if level is not found
    }

    /**
     * @brief Get rank and world size for distributed logging
     *
     * @param rank Initial rank value
     * @param world_size Initial world size value
     * @param auto_detect_env Environment for auto-detection
     * @return std::pair<int, int> The detected rank and world size
     */
    [[nodiscard]] static std::pair<int, int> get_rank_and_world_size(int rank, int world_size, const std::string &auto_detect_env)
    {
        if (rank != -1 && world_size != -1)
            return {rank, world_size};

        static const std::unordered_map<std::string, std::pair<const char *, const char *>> env_vars = {
            {"mpirun", {"OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"}},
            {"torchrun", {"RANK", "WORLD_SIZE"}},
            {"horovod", {"HOROVOD_RANK", "HOROVOD_SIZE"}},
            {"slurm", {"SLURM_PROCID", "SLURM_NTASKS"}},
            {"nccl", {"NCCL_RANK", "NCCL_WORLD_SIZE"}},
            {"general", {"RANK", "WORLD_SIZE"}}};

        auto detect_from_env = [](const char *rank_var, const char *size_var) -> std::optional<std::pair<int, int>>
        {
            if (const char *rank_env = std::getenv(rank_var); rank_env)
            {
                if (const char *size_env = std::getenv(size_var); size_env)
                    return std::make_pair(std::stoi(rank_env), std::stoi(size_env));
            }
            return std::nullopt;
        };

        if (auto_detect_env == "all" || auto_detect_env.empty())
        {
            for (const auto &[_, vars] : env_vars)
            {
                if (auto result = detect_from_env(vars.first, vars.second); result)
                    return *result;
            }
        }
        else if (auto it = env_vars.find(auto_detect_env); it != env_vars.end())
        {
            if (auto result = detect_from_env(it->second.first, it->second.second); result)
                return *result;
        }

        return {0, 1}; // Default values if no detection method succeeds
    }

    /**
     * @brief Log a message to a specific file
     *
     * @param msg The formatted message to log
     * @param file_path The path to the file to log to
     */
    void log_to_file(const std::string &msg, const std::string &file_path)
    {
        fs::create_directories(fs::path(file_path).parent_path());
        std::ofstream file_stream(file_path, std::ios::app);
        if (file_stream.is_open())
        {
            file_stream << msg;
            file_stream.close();
        }
        else
            std::cerr << "Failed to open new file: " << file_path << std::endl;
    }
};

/**
 * @brief Nanobind module definition
 *
 * This function sets up the Python bindings for the CppLogger class.
 */
NB_MODULE(cpplightlog, m)
{
    // Add a docstring for the entire module
    m.doc() = R"pbdoc(
        A C++ based logging module for Python

        This module provides a logging system implemented in C++ with Python bindings.
        It supports file and console logging, rank-based logging for
        distributed systems, and various log levels.

        Key Features:
        - File and console logging
        - Support for distributed systems with rank-based logging
        - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - Automatic environment detection for distributed systems
        - High-performance C++ implementation with Python interface

        Example usage:
            import _lightlog
            
            # Create a logger
            logger = _lightlog.CppLogger("MyLogger", "logfile.log", level=20)
            
            # Log messages
            logger.log("This is an info message", 20)
            logger.log("This is a warning", 30)
            
            # Close the logger
            logger.close()

        For more detailed information on each class and method, use Python's help() function.
    )pbdoc";

    nb::class_<CppLogger>(m, "CppLogger")
        .def(nb::init<const std::string &, const std::string &, const std::string &, int, bool, int, int, const std::string &, int>(),
             nb::arg("name"),
             nb::arg("file_path") = "",
             nb::arg("mode") = "a",
             nb::arg("level") = -1,
             nb::arg("use_rank") = false,
             nb::arg("rank") = 0,
             nb::arg("world_size") = 1,
             nb::arg("auto_detect_env") = "none",
             nb::arg("log_rank") = -1,
             R"pbdoc(
                Initialize a new CppLogger instance.

                Args:
                    name (str): The name of the logger.
                    file_path (str, optional): Path to the log file. Defaults to "".
                    mode (str, optional): File opening mode. Defaults to "a" (append).
                    level (int, optional): Logging level. Defaults to -1 (no change).
                    use_rank (bool, optional): Whether to use rank-based logging. Defaults to False.
                    rank (int, optional): Process rank for distributed logging. Defaults to 0.
                    world_size (int, optional): Total number of processes. Defaults to 1.
                    auto_detect_env (str, optional): Environment for auto-detecting rank and world size. Defaults to "none".
                    log_rank (int, optional): Specific rank to log on. Defaults to -1 (log on all ranks).

                The logging levels are:
                    10: DEBUG
                    20: INFO
                    30: WARNING
                    40: ERROR
                    50: CRITICAL

                For distributed logging, set use_rank to True and provide rank and world_size,
                or set auto_detect_env to automatically detect these from the environment.
            )pbdoc")
        .def("close", &CppLogger::close,
             R"pbdoc(
                 Close the log file if it's open.

                 This method should be called when you're done logging to ensure all data is written
                 and system resources are properly released.
             )pbdoc")
        .def("log", &CppLogger::log,
             nb::arg("msg"),
             nb::arg("level"),
             nb::arg("use_rank") = false,
             nb::arg("new_file") = "",
             R"pbdoc(
                 Log a message with specified level and options.

                 Args:
                     msg (str): The message to log.
                     level (int): The log level for this message.
                     use_rank (bool, optional): Whether to include rank information for this message. Defaults to False.
                     new_file (str, optional): Optional new file to log this message to. Defaults to "".

                 This method checks if the given level is greater than or equal to the logger's level
                 before actually logging the message. If a new_file is specified, the message will be
                 logged to that file instead of the default log file.
             )pbdoc")
        .def("flush", &CppLogger::flush,
             R"pbdoc(
                 Flush the logger, writing any buffered data.

                 This method ensures that all pending log messages are immediately written to the
                 output stream (file or console). It's useful when you need to ensure all logs
                 are written before a potential crash or when you're about to read the log file.
             )pbdoc")
        .def("reconfigure", &CppLogger::reconfigure,
             nb::arg("name") = "",
             nb::arg("file_path") = "",
             nb::arg("mode") = "a",
             nb::arg("level") = -1,
             nb::arg("use_rank") = false, // Changed "none" to false
             nb::arg("rank") = 0,
             nb::arg("world_size") = 1,
             nb::arg("auto_detect_env") = "none",
             nb::arg("log_rank") = -1,
             R"pbdoc(
                Reconfigure the logger with new settings.

                Args:
                    name (str, optional): New logger name. Defaults to "".
                    file_path (str, optional): New path to the log file. Defaults to "".
                    mode (str, optional): New file opening mode. Defaults to "a" (append).
                    level (int, optional): New logging level. Defaults to -1 (no change).
                    use_rank (bool, optional): Whether to use rank-based logging. Defaults to False.
                    rank (int, optional): New process rank for distributed logging. Defaults to 0.
                    world_size (int, optional): New total number of processes. Defaults to 1.
                    auto_detect_env (str, optional): New environment for auto-detecting rank and world size. Defaults to "none".
                    log_rank (int, optional): Specific rank to log on. Defaults to -1 (log on all ranks).

                This method updates the logger's configuration. If a parameter is not provided, the
                corresponding setting will not be changed.

                Note:
                    The logging level is an integer value, where:
                        - 0: NOTSET
                        - 10: DEBUG
                        - 20: INFO
                        - 30: WARNING
                        - 40: ERROR
                        - 50: CRITICAL
            )pbdoc");
}