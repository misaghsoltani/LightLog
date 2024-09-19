import logging
import timeit

import matplotlib.pyplot as plt

from lightlog import Logger

# Configuration
num_iterations = 100_000
num_runs = 10
num_repeats = 10
log_message = "Log message {}"
images_dir = "images"

# LightLog
light_logger = Logger("light_logger", "lightlog.log")

# Built-in logging
py_logger = logging.getLogger("py_logger")
py_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# File handler
file_handler = logging.FileHandler("pylog.log")
file_handler.setFormatter(formatter)
py_logger.addHandler(file_handler)

# Stream handler (for console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
py_logger.addHandler(stream_handler)


# Benchmark
def benchmark_logger(logger):

    def inner():
        for i in range(num_iterations):
            logger.info(log_message.format(i))

    return inner


light_times = timeit.repeat(benchmark_logger(light_logger), number=num_runs, repeat=num_repeats)
light_logger.close()

py_times = timeit.repeat(benchmark_logger(py_logger), number=num_runs, repeat=num_repeats)
py_logger.removeHandler(file_handler)
file_handler.close()
py_logger.removeHandler(stream_handler)

# Print results
print(" Benchmark Results ".center(40, "-"))
print(f"Iterations: {num_iterations}")
print(f"Runs per repeat: {num_runs}")
print(f"Repeats: {num_repeats}")
print(f"LightLog times: {[f'{t:.6f} seconds' for t in light_times]}")
print(f"LightLog average time: {sum(light_times) / num_repeats:.6f} seconds")
print(f"Built-in logging times: {[f'{t:.6f} seconds' for t in py_times]}")
print(f"Built-in logging average time: {sum(py_times) / num_repeats:.6f} seconds")
print("-" * 40)

# Plot results

# Bar chart for individual times
plt.figure(figsize=(8, 6))
plt.bar(range(num_repeats), py_times, label='Built-in Logging', color='#4c516d',
        alpha=0.85)  # Dark grey
plt.bar(range(num_repeats), light_times, label='LightLog', color='#2a9d8f', alpha=0.85)  # Teal
plt.xlabel('Repeat')
plt.ylabel('Time (seconds)')
plt.title('Individual Times')
plt.legend()
plt.tight_layout()
plt.savefig(f'{images_dir}/individual_times.png')

# Bar chart for average times
plt.figure(figsize=(6, 6))
plt.bar(
    ['LightLog', 'Built-in Logging'],
    [sum(light_times) / num_repeats, sum(py_times) / num_repeats],
    color=['#2a9d8f', '#4c516d'],  # Teal and dark grey
    alpha=0.85)
plt.xlabel('Logger')
plt.ylabel('Average Time (seconds)')
plt.title('Average Times')
plt.tight_layout()
plt.savefig(f'{images_dir}/average_times.png')

# Bar chart for individual times with same scale
plt.figure(figsize=(8, 6))
plt.bar(range(num_repeats), py_times, label='Built-in Logging', color='#4c516d',
        alpha=0.85)  # Dark grey
plt.bar(range(num_repeats), light_times, label='LightLog', color='#2a9d8f', alpha=0.85)  # Teal
plt.xlabel('Repeat')
plt.ylabel('Time (seconds)')
plt.title('Individual Times (Same Scale)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{images_dir}/individual_times_same_scale.png')

# Bar chart for average times with same scale
plt.figure(figsize=(6, 6))
plt.bar(
    ['LightLog', 'Built-in Logging'],
    [sum(light_times) / num_repeats, sum(py_times) / num_repeats],
    color=['#2a9d8f', '#4c516d'],  # Teal and dark grey
    alpha=0.85)
plt.xlabel('Logger')
plt.ylabel('Average Time (seconds)')
plt.title('Average Times (Same Scale)')
plt.ylim(0, max(sum(light_times) / num_repeats, sum(py_times) / num_repeats) * 1.1)
plt.tight_layout()
plt.savefig(f'{images_dir}/average_times_same_scale.png')
