import argparse
import torch
import torch.nn as nn
import time
import psutil
import gc
import csv
import os
from datetime import datetime
import numpy as np
from exp.exp_ReTaMTSF_Environment import Exp_Main


class ResourceMonitor:
    """Monitor and record resource consumption during model execution."""

    def __init__(self):
        self.results = {
            'forward_time': [],
            'peak_gpu_memory_mb': [],
            'peak_cpu_memory_mb': [],
        }

    def get_gpu_memory(self):
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return allocated, reserved
        return 0, 0

    def get_cpu_memory(self):
        """Get current CPU (RAM) memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def reset_peak_memory_stats(self):
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

    def get_peak_gpu_memory(self):
        """Get peak GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0


class BenchmarkExperiment(Exp_Main):
    """Extended experiment class for benchmarking resource consumption."""

    def __init__(self, args):
        super().__init__(args)
        self.monitor = ResourceMonitor()

    def benchmark_single_config(self, num_iterations=50, warmup_iterations=10):
        """
        Benchmark resource consumption for a single configuration.

        Returns:
            dict: Benchmark results including forward time, memory usage, GPU hours
        """
        results = {
            'config': {
                'ablation': self.args.ablation,
                'pred_len': self.args.pred_len,  # H
                'nref': self.args.nref,  # K_T
                'nref_text': self.args.nref_text,  # K_n
                'batch_size': self.args.batch_size,
                'seq_len': self.args.seq_len,
                'label_len': self.args.label_len,
            },
            'forward_time_ms': [],
            'gpu_memory_allocated_mb': [],
            'cpu_memory_mb': [],
        }

        # Get a single batch for benchmarking
        test_data, test_loader = self._get_data(flag='test', target_ids=self.args.target_ids)

        # Get first batch
        batch = None
        for batch_data in test_loader:
            batch = batch_data
            break

        if batch is None:
            print("No data available for benchmarking!")
            return results

        (batch_target_series_x, batch_target_series_y,
         batch_TS_database, batch_qt, batch_des, batch_newsdatabase) = batch

        # Move to device
        batch_target_series_x = batch_target_series_x.float().to(self.device)
        batch_TS_database = batch_TS_database.float().to(self.device)
        batch_qt = batch_qt.float().to(self.device)
        batch_des = batch_des.float().to(self.device)
        batch_newsdatabase = batch_newsdatabase.float().to(self.device)

        self.model.eval()

        # Warmup iterations
        print(f"Running {warmup_iterations} warmup iterations...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(batch_target_series_x, batch_TS_database,
                              batch_qt, batch_des, batch_newsdatabase)

        # Synchronize before benchmarking
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Reset peak memory stats
        self.monitor.reset_peak_memory_stats()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Track peak CPU memory during benchmarking
        peak_cpu_memory = 0

        # Benchmark iterations
        print(f"Running {num_iterations} benchmark iterations...")
        with torch.no_grad():
            for i in range(num_iterations):
                # Measure CPU memory before
                cpu_mem_before = self.monitor.get_cpu_memory()

                # Start timing
                start_time = time.perf_counter()

                # Forward pass
                outputs, temp_emb, text_emb, TS, TS_emb = self.model(
                    batch_target_series_x, batch_TS_database,
                    batch_qt, batch_des, batch_newsdatabase
                )

                # End timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                # Record metrics
                forward_time_ms = (end_time - start_time) * 1000
                gpu_alloc, _ = self.monitor.get_gpu_memory()
                cpu_mem_after = self.monitor.get_cpu_memory()

                # Update peak CPU memory
                peak_cpu_memory = max(peak_cpu_memory, cpu_mem_after)

                results['forward_time_ms'].append(forward_time_ms)
                results['gpu_memory_allocated_mb'].append(gpu_alloc)
                results['cpu_memory_mb'].append(cpu_mem_after)

                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{num_iterations}: {forward_time_ms:.2f} ms")

        # Get peak GPU memory
        results['peak_gpu_memory_mb'] = self.monitor.get_peak_gpu_memory()
        results['peak_cpu_memory_mb'] = peak_cpu_memory

        # Calculate statistics
        forward_time_mean_ms = np.mean(results['forward_time_ms'])
        forward_time_std_ms = np.std(results['forward_time_ms'])

        # GPU hours calculation
        # Forward time per batch in seconds
        avg_time_sec = forward_time_mean_ms / 1000
        # Samples per second = batch_size / avg_time_sec
        samples_per_sec = self.args.batch_size / avg_time_sec
        # GPU hours per 1 sample = (1 / samples_per_sec) / 3600
        gpu_hours_per_sample = (1 / samples_per_sec) / 3600
        # GPU hours per 1000 samples
        gpu_hours_per_1k_samples = gpu_hours_per_sample * 1000

        results['stats'] = {
            'forward_time_mean_ms': forward_time_mean_ms,
            'forward_time_std_ms': forward_time_std_ms,
            'forward_time_min_ms': np.min(results['forward_time_ms']),
            'forward_time_max_ms': np.max(results['forward_time_ms']),
            'peak_gpu_memory_mb': results['peak_gpu_memory_mb'],
            'peak_cpu_memory_mb': results['peak_cpu_memory_mb'],
            'samples_per_sec': samples_per_sec,
            'gpu_hours_per_sample': gpu_hours_per_sample,
            'gpu_hours_per_1k_samples': gpu_hours_per_1k_samples,
        }

        return results

    def run_parameter_sweep(self, param_name, param_values, num_iterations=50):
        """
        Run benchmark across different values of a parameter.

        Args:
            param_name: Name of parameter to sweep ('pred_len', 'nref', 'nref_text')
            param_values: List of values to test
            num_iterations: Number of iterations per configuration

        Returns:
            list: All benchmark results (one per parameter value)
        """
        sweep_results = []

        original_value = getattr(self.args, param_name)

        for value in param_values:
            print(f"\n{'='*60}")
            print(f"Testing {param_name} = {value}")
            print(f"{'='*60}")

            # Set parameter value
            setattr(self.args, param_name, value)

            # For sweep_Kn, also update qt_ref_text to match nref_text
            # This ensures K_n = K_n_qt, avoiding negative k in topk
            if param_name == 'nref_text':
                self.args.qt_ref_text = value

            # Rebuild model if needed (for architecture changes)
            self.model = self._build_model().to(self.device)

            # Run benchmark
            results = self.benchmark_single_config(num_iterations=num_iterations)
            sweep_results.append(results)

            # Print summary
            stats = results['stats']
            print(f"\nResults for {param_name}={value}:")
            print(f"  Forward time: {stats['forward_time_mean_ms']:.2f} +/- {stats['forward_time_std_ms']:.2f} ms")
            print(f"  Peak GPU memory: {stats['peak_gpu_memory_mb']:.1f} MB")
            print(f"  Peak CPU (RAM) memory: {stats['peak_cpu_memory_mb']:.1f} MB")
            print(f"  Samples/sec: {stats['samples_per_sec']:.1f}")
            print(f"  GPU hours per sample: {stats['gpu_hours_per_sample']:.8f}")
            print(f"  GPU hours per 1K samples: {stats['gpu_hours_per_1k_samples']:.6f}")

            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore original value
        setattr(self.args, param_name, original_value)

        return sweep_results

    def run_comparative_benchmark(self, num_iterations=50):
        """
        Run comparative benchmark between Unimodal (ablation=1) and Multimodal (ablation=0).

        Returns:
            dict: Comparative results
        """
        comparative_results = {
            'unimodal': None,
            'multimodal': None,
            'overhead': {},
        }

        # Test Unimodal (ablation=1)
        print("\n" + "="*60)
        print("Testing UNIMODAL (ablation=1) - Time Series Only")
        print("="*60)
        self.args.ablation = 1
        self.model = self._build_model().to(self.device)
        comparative_results['unimodal'] = self.benchmark_single_config(num_iterations=num_iterations)

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test Multimodal (ablation=0)
        print("\n" + "="*60)
        print("Testing MULTIMODAL (ablation=0) - Time Series + Text")
        print("="*60)
        self.args.ablation = 0
        self.model = self._build_model().to(self.device)
        comparative_results['multimodal'] = self.benchmark_single_config(num_iterations=num_iterations)

        # Calculate overhead
        uni_stats = comparative_results['unimodal']['stats']
        multi_stats = comparative_results['multimodal']['stats']

        comparative_results['overhead'] = {
            'forward_time_ms': multi_stats['forward_time_mean_ms'] - uni_stats['forward_time_mean_ms'],
            'forward_time_percent': ((multi_stats['forward_time_mean_ms'] / uni_stats['forward_time_mean_ms']) - 1) * 100,
            'peak_gpu_memory_mb': multi_stats['peak_gpu_memory_mb'] - uni_stats['peak_gpu_memory_mb'],
            'peak_gpu_memory_percent': ((multi_stats['peak_gpu_memory_mb'] / max(uni_stats['peak_gpu_memory_mb'], 1)) - 1) * 100,
            'peak_cpu_memory_mb': multi_stats['peak_cpu_memory_mb'] - uni_stats['peak_cpu_memory_mb'],
            'gpu_hours_per_1k_samples': multi_stats['gpu_hours_per_1k_samples'] - uni_stats['gpu_hours_per_1k_samples'],
        }

        return comparative_results


def save_results_to_csv(results, filename):
    """Save benchmark results to CSV file."""
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        if 'unimodal' in results:
            # Comparative benchmark results
            writer.writerow([
                'Mode', 'H (pred_len)', 'K_T (nref)', 'K_n (nref_text)',
                'Forward Time (ms)', 'Time Std (ms)',
                'Peak GPU Memory (MB)', 'Peak CPU Memory (MB)',
                'Samples/sec', 'GPU Hours/1K Samples'
            ])

            for mode in ['unimodal', 'multimodal']:
                r = results[mode]
                config = r['config']
                stats = r['stats']
                mode_name = 'Unimodal' if mode == 'unimodal' else 'Multimodal'

                writer.writerow([
                    mode_name,
                    config['pred_len'],
                    config['nref'],
                    config['nref_text'],
                    f"{stats['forward_time_mean_ms']:.4f}",
                    f"{stats['forward_time_std_ms']:.4f}",
                    f"{stats['peak_gpu_memory_mb']:.2f}",
                    f"{stats['peak_cpu_memory_mb']:.2f}",
                    f"{stats['samples_per_sec']:.2f}",
                    f"{stats['gpu_hours_per_1k_samples']:.8f}",
                ])

            # Write overhead summary
            writer.writerow([])
            writer.writerow(['OVERHEAD (Multimodal - Unimodal)'])
            overhead = results['overhead']
            writer.writerow(['Forward Time (ms)', f"{overhead['forward_time_ms']:.4f}"])
            writer.writerow(['Forward Time (%)', f"{overhead['forward_time_percent']:.2f}"])
            writer.writerow(['Peak GPU Memory (MB)', f"{overhead['peak_gpu_memory_mb']:.2f}"])
            writer.writerow(['Peak GPU Memory (%)', f"{overhead['peak_gpu_memory_percent']:.2f}"])
            writer.writerow(['Peak CPU Memory (MB)', f"{overhead['peak_cpu_memory_mb']:.2f}"])
            writer.writerow(['GPU Hours/1K Samples', f"{overhead['gpu_hours_per_1k_samples']:.8f}"])

        else:
            # Parameter sweep results
            writer.writerow([
                'Mode', 'Sweep Parameter', 'Sweep Value',
                'H (pred_len)', 'K_T (nref)', 'K_n (nref_text)',
                'Forward Time (ms)', 'Time Std (ms)',
                'Peak GPU Memory (MB)', 'Peak CPU Memory (MB)',
                'Samples/sec', 'GPU Hours/1K Samples'
            ])

            for r in results:
                config = r['config']
                stats = r['stats']
                mode_name = 'Unimodal' if config['ablation'] == 1 else 'Multimodal'

                writer.writerow([
                    mode_name,
                    'N/A',
                    'N/A',
                    config['pred_len'],
                    config['nref'],
                    config['nref_text'],
                    f"{stats['forward_time_mean_ms']:.4f}",
                    f"{stats['forward_time_std_ms']:.4f}",
                    f"{stats['peak_gpu_memory_mb']:.2f}",
                    f"{stats['peak_cpu_memory_mb']:.2f}",
                    f"{stats['samples_per_sec']:.2f}",
                    f"{stats['gpu_hours_per_1k_samples']:.8f}",
                ])

    print(f"\nResults saved to {filename}")


def print_comparative_summary(comparative_results):
    """Print a formatted summary of comparative results."""
    print("\n" + "="*100)
    print("COMPARATIVE BENCHMARK SUMMARY: Unimodal vs Multimodal")
    print("="*100)

    uni = comparative_results['unimodal']['stats']
    multi = comparative_results['multimodal']['stats']
    overhead = comparative_results['overhead']
    config = comparative_results['unimodal']['config']

    print(f"\nConfiguration: H={config['pred_len']}, K_T={config['nref']}, K_n={config['nref_text']}, "
          f"batch_size={config['batch_size']}, seq_len={config['seq_len']}, label_len={config['label_len']}")
    print()
    print(f"{'Metric':<35} {'Unimodal':>20} {'Multimodal':>20} {'Overhead':>20}")
    print("-" * 100)

    print(f"{'Forward Time (ms)':<35} {uni['forward_time_mean_ms']:>20.4f} {multi['forward_time_mean_ms']:>20.4f} {overhead['forward_time_ms']:>+19.4f}")
    print(f"{'Forward Time Std (ms)':<35} {uni['forward_time_std_ms']:>20.4f} {multi['forward_time_std_ms']:>20.4f} {'':>20}")
    print(f"{'Peak GPU Memory (MB)':<35} {uni['peak_gpu_memory_mb']:>20.2f} {multi['peak_gpu_memory_mb']:>20.2f} {overhead['peak_gpu_memory_mb']:>+19.2f}")
    print(f"{'Peak CPU Memory (MB)':<35} {uni['peak_cpu_memory_mb']:>20.2f} {multi['peak_cpu_memory_mb']:>20.2f} {overhead['peak_cpu_memory_mb']:>+19.2f}")
    print(f"{'Samples/sec':<35} {uni['samples_per_sec']:>20.2f} {multi['samples_per_sec']:>20.2f} {'':>20}")
    print(f"{'GPU Hours per Sample':<35} {uni['gpu_hours_per_sample']:>20.8f} {multi['gpu_hours_per_sample']:>20.8f} {'':>20}")
    print(f"{'GPU Hours per 1K Samples':<35} {uni['gpu_hours_per_1k_samples']:>20.8f} {multi['gpu_hours_per_1k_samples']:>20.8f} {overhead['gpu_hours_per_1k_samples']:>+19.8f}")

    print("\n" + "="*100)


def print_sweep_summary(sweep_results, param_name):
    """Print a formatted summary of parameter sweep results."""
    print("\n" + "="*120)
    print(f"PARAMETER SWEEP SUMMARY: {param_name}")
    print("="*120)

    mode_name = 'Unimodal' if sweep_results[0]['config']['ablation'] == 1 else 'Multimodal'
    print(f"\nMode: {mode_name}")
    print()
    print(f"{param_name:<10} {'Forward Time(ms)':>18} {'Time Std(ms)':>15} {'Peak GPU(MB)':>15} {'Peak RAM(MB)':>15} {'GPU Hours/1K':>18}")
    print("-" * 120)

    for r in sweep_results:
        config = r['config']
        stats = r['stats']
        value = config[param_name]

        print(f"{value:<10} {stats['forward_time_mean_ms']:>18.4f} {stats['forward_time_std_ms']:>15.4f} "
              f"{stats['peak_gpu_memory_mb']:>15.2f} {stats['peak_cpu_memory_mb']:>15.2f} "
              f"{stats['gpu_hours_per_1k_samples']:>18.8f}")

    print("\n" + "="*120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReTaMTSF Resource Consumption Benchmark')

    # Random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # Basic config
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--ablation', type=int, default=0, help='ablation mode')
    parser.add_argument('--ablation_syn', type=int, default=0, help='TS synthesis ablation mode')

    # Data paths
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path')
    parser.add_argument('--TS_data_path', type=str,
                        default='Time-MMD/numerical/Environment/Environment.parquet',
                        help='Time series data path')
    parser.add_argument('--QT_emb_path', type=str,
                        default='Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint',
                        help='Query text embedding data path')
    parser.add_argument('--Des_emb_path', type=str,
                        default='Time-MMD/textual/Environment/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description',
                        help='Description embedding data path')
    parser.add_argument('--NewsDatabase_path', type=str,
                        default='Time-MMD/textual/Environment/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2',
                        help='News database path')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='model checkpoints path')
    parser.add_argument('--num_data', type=int, default=1, help='number of data points in total')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # ReTaMTSF parameters
    parser.add_argument('--nperseg', type=int, default=48, help='number of samples per segment')
    parser.add_argument('--nref', type=int, default=3, help='number of reference TS (K_T)')
    parser.add_argument('--naggregation', type=int, default=3, help='number of aggregation module')
    parser.add_argument('--nref_text', type=int, default=4, help='number of text retrieval (K_n)')
    parser.add_argument('--qt_ref_text', type=int, default=4, help='number of text retrieval by qt')
    parser.add_argument('--text_mode', type=str, default='retrieval',
                        choices=['retrieval', 'aligned', 'random'],
                        help='text retrieval mode')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='sequence length (L)')
    parser.add_argument('--pred_len', type=int, default=48, help='predicted length (H)')
    parser.add_argument('--label_len', type=int, default=48, help='predicted step')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--target_ids', nargs='+', required=True, type=str, help='names of target TS')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='l2 lambda')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')

    # Benchmark specific
    parser.add_argument('--benchmark_type', type=str, default='comparative',
                        choices=['comparative', 'sweep_H', 'sweep_KT', 'sweep_Kn'],
                        help='Type of benchmark to run')
    parser.add_argument('--sweep_values', type=int, nargs='+', default=None,
                        help='Values to sweep (for sweep benchmarks)')
    parser.add_argument('--num_iterations', type=int, default=50,
                        help='Number of iterations per benchmark configuration')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                        help='Directory to save benchmark results')

    args = parser.parse_args()

    # Setup
    import random
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create experiment
    exp = BenchmarkExperiment(args)

    if args.benchmark_type == 'comparative':
        # Run comparative benchmark (Unimodal vs Multimodal)
        results = exp.run_comparative_benchmark(num_iterations=args.num_iterations)
        print_comparative_summary(results)

        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.output_dir, f'comparative_benchmark_{timestamp}.csv')
        save_results_to_csv(results, filename)

    elif args.benchmark_type in ['sweep_H', 'sweep_KT', 'sweep_Kn']:
        # Determine parameter and values
        if args.benchmark_type == 'sweep_H':
            param_name = 'pred_len'
            default_values = [48, 96, 192, 336]
            # sweep_H: Run both Unimodal and Multimodal
            modes_to_run = [('Unimodal', 1), ('Multimodal', 0)]
        elif args.benchmark_type == 'sweep_KT':
            param_name = 'nref'
            default_values = [1, 2, 3, 4, 5, 6, 7, 8]
            # sweep_KT: Run both Unimodal and Multimodal
            modes_to_run = [('Unimodal', 1), ('Multimodal', 0)]
        else:  # sweep_Kn
            param_name = 'nref_text'
            default_values = [1, 2, 3, 4, 5, 6, 7, 8]
            # sweep_Kn: Only run Multimodal (K_n has no effect in Unimodal)
            modes_to_run = [('Multimodal', 0)]

        param_values = args.sweep_values if args.sweep_values else default_values

        # Run parameter sweep for specified modes
        for mode_name, mode_value in modes_to_run:
            print(f"\n{'#'*80}")
            print(f"# Running {mode_name} Benchmark Sweep")
            print(f"{'#'*80}")

            args.ablation = mode_value
            exp.model = exp._build_model().to(exp.device)

            results = exp.run_parameter_sweep(param_name, param_values,
                                                num_iterations=args.num_iterations)
            print_sweep_summary(results, param_name)

            # Save results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(args.output_dir,
                                    f'{args.benchmark_type}_{mode_name.lower()}_{timestamp}.csv')
            save_results_to_csv(results, filename)

    print("\nBenchmark completed!")
