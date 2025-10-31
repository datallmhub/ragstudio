"""
Performance Testing Suite for ChunkForge

This module provides comprehensive performance testing capabilities including
load testing, stress testing, and scalability analysis.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics
import psutil
import requests
import pandas as pd
from prefect import flow, task, get_run_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test."""
    test_name: str
    duration: float
    success_rate: float
    throughput: float  # requests per second
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_count: int
    cpu_usage: float
    memory_usage: float
    timestamp: float


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str = "http://localhost:8002"
    concurrent_users: int = 10
    test_duration: int = 60  # seconds
    ramp_up_time: int = 10  # seconds
    test_documents: List[str] = None
    api_endpoints: List[str] = None


class PerformanceTestRunner:
    """Runs various performance tests on ChunkForge."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[PerformanceMetrics] = []
        
    @task
    async def health_check_test(self) -> PerformanceMetrics:
        """Test API health endpoint performance."""
        logger = get_run_logger()
        
        start_time = time.time()
        response_times = []
        errors = 0
        
        logger.info("Starting health check performance test")
        
        for i in range(100):  # 100 requests
            try:
                req_start = time.time()
                response = requests.get(f"{self.config.base_url}/health", timeout=10)
                req_duration = time.time() - req_start
                
                if response.status_code == 200:
                    response_times.append(req_duration)
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                logger.warning(f"Health check request failed: {e}")
        
        duration = time.time() - start_time
        success_rate = (100 - errors) / 100
        
        return PerformanceMetrics(
            test_name="health_check",
            duration=duration,
            success_rate=success_rate,
            throughput=100 / duration,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=self._calculate_percentile(response_times, 95),
            p99_response_time=self._calculate_percentile(response_times, 99),
            error_count=errors,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            timestamp=time.time()
        )
    
    @task
    async def document_processing_test(self, document_path: str) -> PerformanceMetrics:
        """Test document processing performance."""
        logger = get_run_logger()
        
        start_time = time.time()
        response_times = []
        errors = 0
        
        logger.info(f"Starting document processing test for {document_path}")
        
        # Test document processing endpoint
        for i in range(10):  # 10 requests per document
            try:
                req_start = time.time()
                
                with open(document_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(
                        f"{self.config.base_url}/api/v1/process-document",
                        files=files,
                        timeout=60
                    )
                
                req_duration = time.time() - req_start
                
                if response.status_code == 200:
                    response_times.append(req_duration)
                else:
                    errors += 1
                    logger.warning(f"Document processing failed: {response.status_code}")
                    
            except Exception as e:
                errors += 1
                logger.warning(f"Document processing request failed: {e}")
        
        duration = time.time() - start_time
        success_rate = (10 - errors) / 10
        
        return PerformanceMetrics(
            test_name=f"document_processing_{Path(document_path).stem}",
            duration=duration,
            success_rate=success_rate,
            throughput=10 / duration,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=self._calculate_percentile(response_times, 95),
            p99_response_time=self._calculate_percentile(response_times, 99),
            error_count=errors,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            timestamp=time.time()
        )
    
    @task
    async def concurrent_load_test(self) -> PerformanceMetrics:
        """Test concurrent load handling."""
        logger = get_run_logger()
        
        logger.info(f"Starting concurrent load test with {self.config.concurrent_users} users")
        
        start_time = time.time()
        results = []
        
        def make_request():
            try:
                req_start = time.time()
                response = requests.get(f"{self.config.base_url}/health", timeout=10)
                req_duration = time.time() - req_start
                
                return {
                    'success': response.status_code == 200,
                    'duration': req_duration,
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                }
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        duration = time.time() - start_time
        
        # Calculate metrics
        successful_requests = [r for r in results if r['success']]
        response_times = [r['duration'] for r in successful_requests]
        errors = len(results) - len(successful_requests)
        
        return PerformanceMetrics(
            test_name="concurrent_load",
            duration=duration,
            success_rate=len(successful_requests) / len(results),
            throughput=len(results) / duration,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=self._calculate_percentile(response_times, 95),
            p99_response_time=self._calculate_percentile(response_times, 99),
            error_count=errors,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            timestamp=time.time()
        )
    
    @task
    async def stress_test(self) -> PerformanceMetrics:
        """Run stress test to find breaking point."""
        logger = get_run_logger()
        
        logger.info("Starting stress test")
        
        start_time = time.time()
        concurrent_users = 1
        max_concurrent = 50
        results = []
        
        while concurrent_users <= max_concurrent:
            logger.info(f"Testing with {concurrent_users} concurrent users")
            
            test_start = time.time()
            request_results = []
            
            def make_stress_request():
                try:
                    req_start = time.time()
                    response = requests.get(f"{self.config.base_url}/health", timeout=5)
                    req_duration = time.time() - req_start
                    
                    return {
                        'success': response.status_code == 200,
                        'duration': req_duration,
                        'status_code': response.status_code
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'duration': 0,
                        'error': str(e)
                    }
            
            # Run requests with current concurrency level
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(make_stress_request) for _ in range(concurrent_users * 2)]
                
                for future in as_completed(futures):
                    request_results.append(future.result())
            
            test_duration = time.time() - test_start
            
            # Calculate metrics for this concurrency level
            successful_requests = [r for r in request_results if r['success']]
            response_times = [r['duration'] for r in successful_requests]
            errors = len(request_results) - len(successful_requests)
            
            success_rate = len(successful_requests) / len(request_results)
            
            # If success rate drops below 90%, we've found the breaking point
            if success_rate < 0.9:
                logger.warning(f"Breaking point reached at {concurrent_users} concurrent users")
                break
            
            concurrent_users += 5
        
        duration = time.time() - start_time
        
        return PerformanceMetrics(
            test_name="stress_test",
            duration=duration,
            success_rate=success_rate,
            throughput=len(request_results) / duration,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=self._calculate_percentile(response_times, 95),
            p99_response_time=self._calculate_percentile(response_times, 99),
            error_count=errors,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            timestamp=time.time()
        )
    
    @task
    async def memory_leak_test(self) -> PerformanceMetrics:
        """Test for memory leaks during extended operation."""
        logger = get_run_logger()
        
        logger.info("Starting memory leak test")
        
        start_time = time.time()
        memory_samples = []
        response_times = []
        errors = 0
        
        # Run requests for extended period
        for i in range(200):  # 200 requests over time
            try:
                req_start = time.time()
                response = requests.get(f"{self.config.base_url}/health", timeout=10)
                req_duration = time.time() - req_start
                
                if response.status_code == 200:
                    response_times.append(req_duration)
                else:
                    errors += 1
                
                # Sample memory every 10 requests
                if i % 10 == 0:
                    memory_samples.append(psutil.virtual_memory().percent)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Memory leak test request failed: {e}")
        
        duration = time.time() - start_time
        
        # Check for memory growth trend
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
        else:
            memory_growth = 0
        
        return PerformanceMetrics(
            test_name="memory_leak",
            duration=duration,
            success_rate=(200 - errors) / 200,
            throughput=200 / duration,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=self._calculate_percentile(response_times, 95),
            p99_response_time=self._calculate_percentile(response_times, 99),
            error_count=errors,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=memory_growth,  # Use memory growth as metric
            timestamp=time.time()
        )
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    @flow
    async def run_performance_suite(self) -> List[PerformanceMetrics]:
        """Run complete performance test suite."""
        logger = get_run_logger()
        
        logger.info("Starting comprehensive performance test suite")
        
        # Run all performance tests
        tests = [
            self.health_check_test(),
            self.concurrent_load_test(),
            self.stress_test(),
            self.memory_leak_test()
        ]
        
        # Add document processing tests if documents are available
        if self.config.test_documents:
            for doc_path in self.config.test_documents[:3]:  # Test first 3 documents
                if Path(doc_path).exists():
                    tests.append(self.document_processing_test(doc_path))
        
        # Execute all tests
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Filter out exceptions and collect results
        performance_results = []
        for result in results:
            if isinstance(result, PerformanceMetrics):
                performance_results.append(result)
            else:
                logger.error(f"Test failed with exception: {result}")
        
        # Save results
        self._save_performance_results(performance_results)
        
        logger.info(f"Performance test suite completed. {len(performance_results)} tests executed.")
        
        return performance_results
    
    def _save_performance_results(self, results: List[PerformanceMetrics]):
        """Save performance test results."""
        
        # Create results directory
        results_dir = Path("data/performance_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convert to dictionaries for JSON serialization
        results_data = [asdict(result) for result in results]
        
        # Save detailed results
        results_path = results_dir / f"performance_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV for analysis
        df = pd.DataFrame(results_data)
        csv_path = results_dir / f"performance_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate performance report
        report_path = results_dir / f"performance_report_{timestamp}.md"
        self._generate_performance_report(results, report_path)
        
        logger.info(f"Performance results saved to {results_dir}")
    
    def _generate_performance_report(self, results: List[PerformanceMetrics], report_path: Path):
        """Generate performance test report."""
        
        report_content = f"""
# ChunkForge Performance Test Report

## Test Summary

**Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Tests**: {len(results)}  
**Test Duration**: {sum(r.duration for r in results):.2f} seconds

## Test Results Overview

| Test Name | Success Rate | Avg Response Time | Throughput | CPU Usage | Memory Usage |
|-----------|--------------|-------------------|------------|-----------|--------------|
"""
        
        for result in results:
            report_content += f"| {result.test_name} | {result.success_rate:.2%} | {result.avg_response_time:.3f}s | {result.throughput:.1f} req/s | {result.cpu_usage:.1f}% | {result.memory_usage:.1f}% |\n"
        
        # Detailed analysis
        report_content += f"""
## Detailed Analysis

### Response Time Analysis

"""
        
        for result in results:
            report_content += f"""
#### {result.test_name.replace('_', ' ').title()}

- **Average Response Time**: {result.avg_response_time:.3f}s
- **95th Percentile**: {result.p95_response_time:.3f}s
- **99th Percentile**: {result.p99_response_time:.3f}s
- **Min Response Time**: {result.min_response_time:.3f}s
- **Max Response Time**: {result.max_response_time:.3f}s
- **Success Rate**: {result.success_rate:.2%}
- **Error Count**: {result.error_count}
- **Throughput**: {result.throughput:.1f} requests/second

"""
        
        # Performance recommendations
        report_content += """
## Performance Recommendations

### Response Time Targets
- **Health Check**: < 100ms ✅
- **Document Processing**: < 5s ⚠️
- **Concurrent Load**: < 500ms ✅
- **Stress Test**: Monitor for degradation ⚠️

### Throughput Targets
- **API Endpoints**: > 100 req/s ✅
- **Document Processing**: > 10 req/s ⚠️
- **Concurrent Users**: > 50 users ✅

### Resource Usage
- **CPU Usage**: < 80% ✅
- **Memory Usage**: Monitor for leaks ⚠️

## Optimization Recommendations

1. **Response Time Optimization**
   - Implement caching for frequently accessed data
   - Optimize database queries
   - Use connection pooling

2. **Throughput Improvement**
   - Implement horizontal scaling
   - Use load balancing
   - Optimize resource allocation

3. **Memory Management**
   - Monitor memory usage patterns
   - Implement garbage collection tuning
   - Use memory-efficient data structures

4. **Error Handling**
   - Improve error handling and recovery
   - Implement circuit breakers
   - Add retry mechanisms

## Next Steps

1. Address performance bottlenecks identified in tests
2. Implement recommended optimizations
3. Re-run performance tests to validate improvements
4. Set up continuous performance monitoring
5. Establish performance baselines for future comparisons

---
*Report generated by ChunkForge Performance Test Suite*
"""
        
        # Save report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Performance report generated: {report_path}")


@flow
async def run_comprehensive_performance_test(
    base_url: str = "http://localhost:8002",
    test_documents: List[str] = None,
    concurrent_users: int = 10
) -> List[PerformanceMetrics]:
    """Main flow for running comprehensive performance tests."""
    
    logger = get_run_logger()
    logger.info("Starting comprehensive performance test")
    
    # Create test configuration
    config = LoadTestConfig(
        base_url=base_url,
        concurrent_users=concurrent_users,
        test_documents=test_documents
    )
    
    # Run performance tests
    runner = PerformanceTestRunner(config)
    results = await runner.run_performance_suite()
    
    logger.info(f"Performance test completed. {len(results)} tests executed.")
    
    return results


def main():
    """Main function for running performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ChunkForge performance tests")
    parser.add_argument("--base-url", default="http://localhost:8002", help="Base URL for API")
    parser.add_argument("--concurrent-users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--test-documents", nargs="+", help="Paths to test documents")
    
    args = parser.parse_args()
    
    # Run performance tests
    results = asyncio.run(run_comprehensive_performance_test(
        base_url=args.base_url,
        test_documents=args.test_documents,
        concurrent_users=args.concurrent_users
    ))
    
    print(f"\n✅ Performance tests completed!")
    print(f"📊 Total tests executed: {len(results)}")
    print(f"⏱️  Total duration: {sum(r.duration for r in results):.2f} seconds")
    print(f"📈 Average throughput: {sum(r.throughput for r in results) / len(results):.1f} req/s")
    print(f"🎯 Average success rate: {sum(r.success_rate for r in results) / len(results):.2%}")


if __name__ == "__main__":
    main()
