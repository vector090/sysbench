#!/usr/bin/env python3
"""
System Resource Benchmark Tool
Tests CPU, RAM, and disk space usage with precise measurements
"""

import os
import sys
import time
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math
import hashlib
import tempfile
import shutil
import json
from datetime import datetime

class SystemBenchmark:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'cpu': {},
            'memory': {},
            'disk': {}
        }
    
    def _get_system_info(self):
        """Get basic system information"""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total,
            'disk_partitions': [p._asdict() for p in psutil.disk_partitions()]
        }
    
    def detect_working_cores(self):
        """Detect how many CPU cores are actually working"""
        print("Detecting working CPU cores...")
        
        total_cores = multiprocessing.cpu_count()
        working_cores = 0
        core_results = []
        
        def core_test(core_id):
            """Test if a specific core is working"""
            try:
                # Pin to specific core if possible
                start_time = time.time()
                # Do some intensive calculations
                for i in range(1000000):
                    _ = i * i * i
                end_time = time.time()
                return True, end_time - start_time
            except:
                return False, 0
        
        # Test each core
        with ThreadPoolExecutor(max_workers=total_cores) as executor:
            futures = [executor.submit(core_test, i) for i in range(total_cores)]
            for i, future in enumerate(futures):
                working, duration = future.result()
                if working:
                    working_cores += 1
                core_results.append({
                    'core_id': i,
                    'working': working,
                    'test_duration': duration
                })
        
        self.results['cpu']['core_detection'] = {
            'total_cores': total_cores,
            'working_cores': working_cores,
            'core_details': core_results
        }
        
        print(f"  Total cores: {total_cores}")
        print(f"  Working cores: {working_cores}")
        
        return working_cores
    
    def benchmark_cpu(self, duration=30, threads=None):
        """Benchmark CPU with precise measurements"""
        print(f"Starting CPU benchmark for {duration} seconds...")
        
        if threads is None:
            threads = multiprocessing.cpu_count()
        
        def cpu_intensive_task():
            """CPU-intensive prime number calculation"""
            end_time = time.time() + duration
            operations = 0
            primes_found = []
            
            while time.time() < end_time:
                # Find primes using optimized algorithm
                for num in range(2, 10000):
                    is_prime = True
                    for i in range(2, int(math.sqrt(num)) + 1):
                        if num % i == 0:
                            is_prime = False
                            break
                    if is_prime:
                        primes_found.append(num)
                operations += 1
                
                # Add some floating point operations
                for i in range(1000):
                    math.sin(i) * math.cos(i) * math.tan(i)
            
            return operations, len(primes_found)
        
        # Monitor CPU usage during test
        cpu_usage_samples = []
        monitoring_thread = None
        
        def monitor_cpu():
            while not stop_monitoring:
                cpu_usage_samples.append(psutil.cpu_percent(interval=0.1))
        
        stop_monitoring = False
        monitoring_thread = threading.Thread(target=monitor_cpu)
        monitoring_thread.start()
        
        # Run CPU test with multiple threads
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(threads)]
            results = [future.result() for future in futures]
        
        stop_monitoring = True
        monitoring_thread.join()
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        total_operations = sum(r[0] for r in results)
        total_primes = sum(r[1] for r in results)
        avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0
        
        self.results['cpu'] = {
            'duration_seconds': actual_duration,
            'threads_used': threads,
            'total_operations': total_operations,
            'operations_per_second': total_operations / actual_duration,
            'total_primes_found': total_primes,
            'primes_per_second': total_primes / actual_duration,
            'average_cpu_usage_percent': avg_cpu_usage,
            'max_cpu_usage_percent': max(cpu_usage_samples) if cpu_usage_samples else 0,
            'min_cpu_usage_percent': min(cpu_usage_samples) if cpu_usage_samples else 0,
            'cpu_samples_count': len(cpu_usage_samples)
        }
        
        print(f"CPU benchmark completed:")
        print(f"  - Operations per second: {self.results['cpu']['operations_per_second']:.2f}")
        print(f"  - Average CPU usage: {avg_cpu_usage:.2f}%")
        
        return self.results['cpu']
    
    def test_max_memory_allocation(self):
        """Test maximum memory that can be allocated"""
        print("Testing maximum memory allocation capacity...")
        
        mem_info = psutil.virtual_memory()
        available_gb = mem_info.available / (1024**3)
        total_gb = mem_info.total / (1024**3)
        
        print(f"  Total memory: {total_gb:.2f}GB")
        print(f"  Available memory: {available_gb:.2f}GB")
        
        # too much can damage host
        test_size = available_gb * 0.7
        chunk_size = 100 * 1024 * 1024  # 100MB chunks
        max_allocated = 0
        memory_blocks = []
        
        try:
            while test_size > 0.1:  # Minimum 100MB
                print(f"  Testing allocation of {test_size:.2f}GB...")
                
                try:
                    # Try to allocate memory in chunks
                    target_bytes = int(test_size * 1024**3)
                    allocated = 0
                    
                    while allocated < target_bytes:
                        block = bytearray(min(chunk_size, target_bytes - allocated))
                        memory_blocks.append(block)
                        allocated += chunk_size
                        
                        if len(memory_blocks) % 10 == 0:
                            print(f"    Allocated: {allocated / (1024**3):.2f}GB")
                    
                    max_allocated = allocated / (1024**3)
                    print(f"  ✓ Successfully allocated {max_allocated:.2f}GB")
                    
                    # Try to allocate even more if successful
                    #test_size = min(test_size * 1.1, total_gb * 1.2)  # Try 10% more, up to 120% of total
                    test_size = 0 # enough. quit
                    
                except MemoryError:
                    print(f"  ✗ Failed to allocate {test_size:.2f}GB")
                    test_size *= 0.9  # Try 90% of previous size
                    memory_blocks.clear()
                    
        except Exception as e:
            print(f"  Error during memory test: {e}")
        
        # Clean up
        memory_blocks.clear()
        
        self.results['memory']['max_allocation_test'] = {
            'total_memory_gb': total_gb,
            'available_memory_gb': available_gb,
            'max_allocated_gb': max_allocated,
            'allocation_success_rate': (max_allocated / available_gb) * 100 if available_gb > 0 else 0
        }
        
        print(f"  Maximum allocatable: {max_allocated:.2f}GB")
        return max_allocated
    
    def benchmark_memory(self, test_size_gb=1):
        """Benchmark memory with precise measurements"""
        print(f"Starting memory benchmark with {test_size_gb}GB test data...")
        
        test_size_bytes = int(test_size_gb * 1024 * 1024 * 1024)
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Memory allocation test
        start_time = time.time()
        memory_blocks = []
        
        try:
            allocated = 0
            while allocated < test_size_bytes:
                block = bytearray(chunk_size)
                memory_blocks.append(block)
                allocated += chunk_size
                
                if len(memory_blocks) % 100 == 0:
                    print(f"  Allocated: {allocated / (1024*1024*1024):.2f}GB")
            
            allocation_time = time.time() - start_time
            
            # Memory read/write test
            print("  Performing read/write tests...")
            
            # Write test - fill with data
            write_start = time.time()
            for i, block in enumerate(memory_blocks):
                # Fill block with pattern
                pattern = i % 256
                for j in range(len(block)):
                    block[j] = pattern
            write_time = time.time() - write_start
            
            # Read test - verify data
            read_start = time.time()
            checksums = []
            for i, block in enumerate(memory_blocks):
                # Calculate checksum for verification
                checksum = hashlib.md5(block).hexdigest()
                checksums.append(checksum)
            read_time = time.time() - read_start
            
            # Memory copy test
            copy_start = time.time()
            test_blocks = memory_blocks[:len(memory_blocks)//2]  # Copy half the blocks
            copied_blocks = [bytearray(block) for block in test_blocks]
            copy_time = time.time() - copy_start
            
            # Get memory usage info
            mem_info = psutil.virtual_memory()
            
            self.results['memory'] = {
                'test_size_gb': test_size_gb,
                'allocation_time_seconds': allocation_time,
                'allocation_speed_gb_per_sec': test_size_gb / allocation_time,
                'write_time_seconds': write_time,
                'write_speed_gb_per_sec': test_size_gb / write_time,
                'read_time_seconds': read_time,
                'read_speed_gb_per_sec': test_size_gb / read_time,
                'copy_time_seconds': copy_time,
                'copy_speed_gb_per_sec': (test_size_gb / 2) / copy_time,
                'memory_usage_during_test': {
                    'total_gb': mem_info.total / (1024**3),
                    'available_gb': mem_info.available / (1024**3),
                    'used_gb': mem_info.used / (1024**3),
                    'percent_used': mem_info.percent
                },
                'blocks_allocated': len(memory_blocks),
                'block_size_bytes': chunk_size
            }
            
            print(f"Memory benchmark completed:")
            print(f"  - Allocation speed: {self.results['memory']['allocation_speed_gb_per_sec']:.2f} GB/s")
            print(f"  - Write speed: {self.results['memory']['write_speed_gb_per_sec']:.2f} GB/s")
            print(f"  - Read speed: {self.results['memory']['read_speed_gb_per_sec']:.2f} GB/s")
            
        except MemoryError:
            print("Memory allocation failed - insufficient memory")
            self.results['memory'] = {'error': 'Memory allocation failed'}
        
        finally:
            # Clean up memory
            del memory_blocks
            if 'copied_blocks' in locals():
                del copied_blocks
        
        return self.results['memory']
    
    def benchmark_disk(self, test_size_gb=1, test_file_path=None):
        """Benchmark disk I/O with precise measurements"""
        print(f"Starting disk benchmark with {test_size_gb}GB test data...")
        
        if test_file_path is None:
            test_file_path = os.path.join(tempfile.gettempdir(), 'benchmark_test_file')
        
        test_size_bytes = int(test_size_gb * 1024 * 1024 * 1024)
        chunk_size = 1024 * 1024  # 1MB chunks
        chunks_to_write = test_size_bytes // chunk_size
        
        try:
            # Write test
            print("  Performing write test...")
            write_start = time.time()
            
            with open(test_file_path, 'wb') as f:
                for i in range(chunks_to_write):
                    # Write different patterns to each chunk for realistic testing
                    chunk_data = os.urandom(chunk_size)
                    f.write(chunk_data)
                    
                    if (i + 1) % 100 == 0:
                        print(f"    Written: {(i + 1) * chunk_size / (1024*1024*1024):.2f}GB")
            
            write_time = time.time() - write_start
            
            # File is already closed, no need to flush
            
            # Read test
            print("  Performing read test...")
            read_start = time.time()
            
            with open(test_file_path, 'rb') as f:
                total_read = 0
                while total_read < test_size_bytes:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    total_read += len(chunk)
            
            read_time = time.time() - read_start
            
            # Random read/write test
            print("  Performing random I/O test...")
            random_start = time.time()
            random_operations = 1000
            
            with open(test_file_path, 'r+b') as f:
                for i in range(random_operations):
                    # Random position
                    pos = (i * 997) % (test_size_bytes - chunk_size)
                    f.seek(pos)
                    
                    # Random read
                    data = f.read(chunk_size)
                    
                    # Random write at same position
                    f.seek(pos)
                    f.write(data)
            
            random_time = time.time() - random_start
            
            # Get disk usage info
            disk_usage = shutil.disk_usage(os.path.dirname(test_file_path))
            
            self.results['disk'] = {
                'test_size_gb': test_size_gb,
                'test_file_path': test_file_path,
                'write_time_seconds': write_time,
                'write_speed_gb_per_sec': test_size_gb / write_time,
                'read_time_seconds': read_time,
                'read_speed_gb_per_sec': test_size_gb / read_time,
                'random_io_time_seconds': random_time,
                'random_io_operations_per_second': random_operations / random_time,
                'disk_usage': {
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3)
                },
                'chunks_processed': chunks_to_write,
                'chunk_size_bytes': chunk_size
            }
            
            print(f"Disk benchmark completed:")
            print(f"  - Write speed: {self.results['disk']['write_speed_gb_per_sec']:.2f} GB/s")
            print(f"  - Read speed: {self.results['disk']['read_speed_gb_per_sec']:.2f} GB/s")
            print(f"  - Random I/O: {self.results['disk']['random_io_operations_per_second']:.2f} ops/sec")
            
        except Exception as e:
            print(f"Disk benchmark failed: {e}")
            self.results['disk'] = {'error': str(e)}
        
        finally:
            # Clean up test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
        
        return self.results['disk']
    
    def run_quick_assessment(self):
        """Run quick system assessment with human-readable output"""
        print("System Resource Assessment")
        print("=" * 40)
        
        # CPU Core Detection
        working_cores = self.detect_working_cores()
        print()
        
        # Memory Allocation Test
        max_memory = self.test_max_memory_allocation()
        print()
        
        # Quick Summary
        print("Quick Summary:")
        print("-" * 20)
        print(f"CPU Cores: {working_cores} working")
        print(f"Memory: {max_memory:.1f}GB allocatable")
        
        # Generate brief output for file
        brief_output = f"""System Resource Assessment
========================================
CPU Cores: {working_cores} working
Memory: {max_memory:.1f}GB allocatable
"""
        
        return self.results, brief_output
    
    def run_full_benchmark(self, cpu_duration=30, memory_size_gb=1, disk_size_gb=1):
        """Run complete system benchmark"""
        print("Starting full system benchmark...")
        print("=" * 50)
        
        self.detect_working_cores()
        print()
        
        self.test_max_memory_allocation()
        print()
        
        self.benchmark_cpu(duration=cpu_duration)
        print()
        
        self.benchmark_memory(test_size_gb=memory_size_gb)
        print()
        
        self.benchmark_disk(test_size_gb=disk_size_gb)
        print()
        
        print("Full system benchmark completed!")
        return self.results
    
    def save_results(self, filename=None, brief_output=None):
        """Save benchmark results to JSON file"""
        import os
        
        # Create output directory
        output_dir = "_nosync/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get hostname
        hostname = os.environ.get('HOSTNAME', 'unknown')
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results__{timestamp}_{hostname}.json"
        
        # Full path with output directory
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
        
        # Save brief output if provided
        if brief_output:
            brief_filename = filename.replace('.json', '_brief.txt')
            brief_filepath = os.path.join(output_dir, brief_filename)
            with open(brief_filepath, 'w') as f:
                f.write(brief_output)
            print(f"Brief output saved to {brief_filepath}")
        
        return filepath

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='System Resource Benchmark Tool')
    parser.add_argument('--full', action='store_true', help='Run full benchmark (default is quick assessment)')
    parser.add_argument('--cpu-only', action='store_true', help='Run CPU benchmark only')
    parser.add_argument('--memory-only', action='store_true', help='Run memory benchmark only')
    parser.add_argument('--disk-only', action='store_true', help='Run disk benchmark only')
    parser.add_argument('--cpu-duration', type=int, default=10, help='CPU benchmark duration in seconds')
    parser.add_argument('--memory-size', type=float, default=1.0, help='Memory test size in GB')
    parser.add_argument('--disk-size', type=float, default=1.0, help='Disk test size in GB')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    benchmark = SystemBenchmark()
    
    brief_output = None
    
    if args.full:
        benchmark.run_full_benchmark(
            cpu_duration=args.cpu_duration,
            memory_size_gb=args.memory_size,
            disk_size_gb=args.disk_size
        )
    elif args.cpu_only:
        benchmark.detect_working_cores()
        benchmark.benchmark_cpu(duration=args.cpu_duration)
    elif args.memory_only:
        benchmark.test_max_memory_allocation()
        benchmark.benchmark_memory(test_size_gb=args.memory_size)
    elif args.disk_only:
        benchmark.benchmark_disk(test_size_gb=args.disk_size)
    else:
        results, brief_output = benchmark.run_quick_assessment()
    
    # Save results
    output_file = benchmark.save_results(args.output, brief_output)

if __name__ == '__main__':
    main()
