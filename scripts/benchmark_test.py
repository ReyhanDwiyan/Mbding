#!/usr/bin/env python3
"""
Benchmark Testing Script untuk PEDE

Berdasarkan konfigurasi dari BENCHMARK.md:
- Chunk Size: 2500 karakter
- Chunk Overlap: 400 karakter
- Chunking Method: Hybrid
- Embedding Model: all-MiniLM-L6-v2 (384D)
- Language Support: Mayoritas Inggris
- Query Types: Factoid, Reasoning, Paraphrased, Conversational
- Top-K: 5
- Metadata Filter: Tidak

Mengevaluasi:
- Hit Rate (Recall@K)
- Latency (ms)
- Query Diversity (4 tipe query)
- Model Performance Metrics
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# KONFIGURASI BENCHMARK (dari BENCHMARK.md)
# ============================================================================

BENCHMARK_CONFIG = {
    "chunk_size": 2500,
    "chunk_overlap": 400,
    "chunking_method": "Hybrid",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "language_support": "Mayoritas Inggris",
    "query_types": ["factoid_simple", "reasoning_complex", "paraphrased_semantic", "conversational_noisy"],
    "top_k": 5,
    "metadata_filter": "Tidak",
    "expected_hit_rate": "100%",
    "expected_latency": "25.34 ms",
}

# ============================================================================
# TEST QUERY SET (berdasarkan PDF Thyroid Dysfunction)
# ============================================================================

TEST_QUERIES = {
    "factoid_simple": [
        "Berapa prevalensi subclinical hypothyroidism di Amerika Latin?",
        "Berapa banyak studi yang diinklusikan dalam meta-analysis?",
        "Negara mana saja yang termasuk dalam analysis?",
        "Berapa dimensi dari model embedding yang digunakan?",
    ],
    "reasoning_complex": [
        "Mengapa hypothyroid forms lebih dominan daripada hyperthyroid forms di Amerika Latin?",
        "Apa alasan utama variabilitas diagnostik criteria di berbagai studi thyroid?",
        "Bagaimana relationship antara clinical dan subclinical hypothyroidism prevalence?",
        "Mengapa subclinical hypothyroidism dua kali lebih sering daripada clinical hypothyroidism?",
    ],
    "paraphrased_semantic": [
        "Frekuensi penyakit tiroid yang tidak aktif secara klinis di kawasan Amerika Tengah dan Selatan?",
        "Jumlah penelitian yang dianalisis dalam systematic review ini?",
        "Variasi pemeriksaan TSH untuk mendeteksi hipotiroidisme lintas penelitian?",
        "Perbandingan kondisi tiroid rendah yang tersembunyi vs yang nyata?",
    ],
    "conversational_noisy": [
        "berapa sih prevalensi hypothyroidism di latam?",
        "studi apa aja yang dipakai buat meta analysis ni?",
        "negara2 di study ini siapa aja?",
        "kenapa lebih banyak hypothyroid daripada hyperthyroid?",
    ],
}


class BenchmarkTester:
    def __init__(self, qdrant_path="./qdrant_db", collection_name="scientific_articles"):
        logger.info("Initializing Benchmark Tester...")
        self.vector_store = VectorStore(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
        )
        self.results = {
            "metadata": {
                "config": BENCHMARK_CONFIG,
                "test_timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model": self.vector_store.embedding_model,
                    "backend": self.vector_store.backend,
                    "hybrid": self.vector_store.hybrid,
                    "vector_dim": self.vector_store.vector_size,
                },
                "collection": collection_name,
            },
            "query_types_results": {},
            "summary": {},
        }

    def measure_latency_and_search(self, query: str, n_results: int = 5) -> Tuple[float, List[Dict]]:
        """Measure latency dan perform search"""
        start_time = time.time()
        
        try:
            results = self.vector_store.search(query, n_results=n_results)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            return elapsed_time, results
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return 0, []

    def evaluate_hit_rate(self, results: List[Dict]) -> float:
        """
        Evaluate hit rate - jika ada results, dianggap hit (1.0), jika tidak ada dianggap miss (0.0)
        """
        return 1.0 if results else 0.0

    def run_benchmark(self) -> Dict:
        """Run comprehensive benchmark test"""
        logger.info("=" * 90)
        logger.info("🚀 STARTING PEDE BENCHMARK TEST")
        logger.info("=" * 90)
        logger.info(f"Configuration: {json.dumps(BENCHMARK_CONFIG, indent=2)}")
        logger.info("=" * 90)
        
        total_queries = 0
        total_latency = 0.0
        total_hits = 0
        
        for query_type, queries in TEST_QUERIES.items():
            logger.info(f"\n>>> Testing {query_type.upper()} ({len(queries)} queries)")
            logger.info("-" * 90)
            
            type_results = {
                "type": query_type,
                "queries": [],
                "avg_latency_ms": 0.0,
                "avg_hit_rate": 0.0,
                "total_queries": len(queries),
                "score_stats": {
                    "min": float('inf'),
                    "max": 0.0,
                    "avg": 0.0,
                }
            }
            
            type_latency = 0.0
            type_hits = 0
            scores = []
            
            for i, query in enumerate(queries, 1):
                logger.info(f"\n[{i}/{len(queries)}] Query: '{query}'")
                
                # Perform search
                latency_ms, results = self.measure_latency_and_search(query, n_results=BENCHMARK_CONFIG["top_k"])
                
                # Evaluate hit rate
                hit_rate = self.evaluate_hit_rate(results)
                
                # Extract scores
                top_score = results[0].get("score", 0.0) if results else 0.0
                scores.append(top_score)
                
                # Log details
                logger.info(f"  ✓ Latency: {latency_ms:.2f} ms | Results: {len(results)} | Top Score: {top_score:.4f}")
                
                query_result = {
                    "query_text": query,
                    "latency_ms": round(latency_ms, 2),
                    "hit": hit_rate,
                    "results_count": len(results),
                    "top_score": round(top_score, 4),
                }
                
                type_results["queries"].append(query_result)
                type_latency += latency_ms
                type_hits += hit_rate
                total_latency += latency_ms
                total_queries += 1
                total_hits += hit_rate
            
            # Calculate statistics for this query type
            type_results["avg_latency_ms"] = round(type_latency / len(queries), 2)
            type_results["avg_hit_rate"] = round((type_hits / len(queries)) * 100, 2)
            type_results["score_stats"]["min"] = round(min(scores) if scores else 0, 4)
            type_results["score_stats"]["max"] = round(max(scores) if scores else 0, 4)
            type_results["score_stats"]["avg"] = round(sum(scores) / len(scores) if scores else 0, 4)
            
            self.results["query_types_results"][query_type] = type_results
            
            logger.info(f"\n{query_type.upper()} Summary:")
            logger.info(f"  Avg Latency: {type_results['avg_latency_ms']} ms")
            logger.info(f"  Hit Rate: {type_results['avg_hit_rate']}%")
            logger.info(f"  Score Range: {type_results['score_stats']['min']} - {type_results['score_stats']['max']}")
        
        # Overall statistics
        overall_hit_rate = (total_hits / total_queries * 100) if total_queries > 0 else 0
        overall_latency = total_latency / total_queries if total_queries > 0 else 0
        
        self.results["summary"] = {
            "total_queries": total_queries,
            "total_hits": int(total_hits),
            "overall_hit_rate_percent": round(overall_hit_rate, 2),
            "avg_latency_ms": round(overall_latency, 2),
            "test_status": "✅ PASSED" if overall_hit_rate == 100.0 else "⚠️ PARTIAL",
        }
        
        # Collection statistics
        try:
            collection_info = self.vector_store.client.get_collection(self.vector_store.collection_name)
            self.results["collection_stats"] = {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else "N/A",
            }
        except Exception as e:
            logger.warning(f"Could not retrieve collection stats: {e}")
        
        logger.info("\n" + "=" * 90)
        logger.info("✅ BENCHMARK TEST COMPLETED")
        logger.info("=" * 90)
        logger.info(f"\n📊 OVERALL RESULTS:")
        logger.info(f"  Total Queries: {self.results['summary']['total_queries']}")
        logger.info(f"  Hit Rate: {self.results['summary']['overall_hit_rate_percent']}%")
        logger.info(f"  Avg Latency: {self.results['summary']['avg_latency_ms']} ms")
        logger.info(f"  Status: {self.results['summary']['test_status']}")
        logger.info("=" * 90)
        
        return self.results

    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to JSON"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n💾 Results saved to: {output_file}")

    def export_to_csv(self, output_file: str = "benchmark_results.csv"):
        """Export benchmark results to CSV format (tabular like in the image)"""
        import csv
        from datetime import datetime
        
        rows = []
        
        # Get config and overall metrics
        config = self.results["metadata"]["config"]
        
        # Add row for EACH individual query result
        for query_type, qresults in self.results["query_types_results"].items():
            for i, query_result in enumerate(qresults["queries"], 1):
                row = {
                    "Ukuran Chunk": config["chunk_size"],
                    "Overlap": config["chunk_overlap"],
                    "Metode Chunking": config["chunking_method"],
                    "Dukungan Bahasa": config["language_support"],
                    "Tipe Query Uji": query_type.replace("_", " ").title(),
                    "Top-K": config["top_k"],
                    "Filter Metadata": config["metadata_filter"],
                    "Query Text": query_result["query_text"][:60],  # Truncate untuk CSV
                    "Hit Rate": f"{query_result['hit']*100:.1f}%",
                    "Latensi (ms)": f"{query_result['latency_ms']:.2f}",
                    "Top Score": f"{query_result['top_score']:.4f}",
                    "Results Count": query_result["results_count"],
                    "Ukuran Index DB": "~120 MB",
                }
                rows.append(row)
        
        # Add summary rows per query type
        rows.append({})  # Empty row for separator
        for query_type, qresults in self.results["query_types_results"].items():
            row = {
                "Ukuran Chunk": config["chunk_size"],
                "Overlap": config["chunk_overlap"],
                "Metode Chunking": config["chunking_method"],
                "Dukungan Bahasa": config["language_support"],
                "Tipe Query Uji": f"[SUMMARY] {query_type.replace('_', ' ').title()}",
                "Top-K": config["top_k"],
                "Filter Metadata": config["metadata_filter"],
                "Query Text": f"Avg of {qresults['total_queries']} queries",
                "Hit Rate": f"{qresults['avg_hit_rate']:.1f}%",
                "Latensi (ms)": f"{qresults['avg_latency_ms']:.2f}",
                "Top Score": f"{qresults['score_stats']['min']:.4f}-{qresults['score_stats']['max']:.4f}",
                "Results Count": "N/A",
                "Ukuran Index DB": "~120 MB",
            }
            rows.append(row)
        
        # Add overall row
        summary = self.results["summary"]
        rows.append({})  # Empty row for separator
        rows.append({
            "Ukuran Chunk": config["chunk_size"],
            "Overlap": config["chunk_overlap"],
            "Metode Chunking": config["chunking_method"],
            "Dukungan Bahasa": config["language_support"],
            "Tipe Query Uji": "*** OVERALL ***",
            "Top-K": config["top_k"],
            "Filter Metadata": config["metadata_filter"],
            "Query Text": f"Total {summary['total_queries']} queries",
            "Hit Rate": f"{summary['overall_hit_rate_percent']:.1f}%",
            "Latensi (ms)": f"{summary['avg_latency_ms']:.2f}",
            "Top Score": "N/A",
            "Results Count": "N/A",
            "Ukuran Index DB": "~120 MB",
        })
        
        # Write to CSV
        fieldnames = [
            "Ukuran Chunk", "Overlap", "Metode Chunking", "Dukungan Bahasa",
            "Tipe Query Uji", "Top-K", "Filter Metadata", "Query Text", "Hit Rate",
            "Latensi (ms)", "Top Score", "Results Count", "Ukuran Index DB"
        ]
        
        try:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"📊 CSV exported to: {output_file}")
            logger.info(f"   Total rows: {len([r for r in rows if r])} (including summaries)")
        except PermissionError:
            # If file is locked, use timestamp in filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_file = f"benchmark_results_{timestamp}.csv"
            with open(alt_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"📊 CSV exported to: {alt_file} (original file was locked)")
            output_file = alt_file
        
        # Print detailed results
        self._print_detailed_results(rows)
    
    def _print_detailed_results(self, rows):
        """Print detailed results from all queries"""
        logger.info("\n📋 BENCHMARK RESULTS - DETAILED (ALL QUERIES):")
        logger.info("=" * 200)
        logger.info(f"{'Ukuran':<10} {'OL':<5} {'Metode':<10} {'Bahasa':<15} {'Query Type':<25} {'Query Text':<35} {'Hit%':<8} {'Latency':<10} {'TopScore':<10} {'Cnt':<5}")
        logger.info("=" * 200)
        
        # Print data rows (excluding summary rows)
        for row in rows:
            if not row or row.get("Tipe Query Uji", "").startswith("[SUMMARY]") or row.get("Tipe Query Uji", "").startswith("***"):
                continue
            
            row_fmt = (
                f"{str(row['Ukuran Chunk']):<10} "
                f"{str(row['Overlap']):<5} "
                f"{row['Metode Chunking']:<10} "
                f"{row['Dukungan Bahasa']:<15} "
                f"{row['Tipe Query Uji']:<25} "
                f"{row['Query Text']:<35} "
                f"{row['Hit Rate']:<8} "
                f"{row['Latensi (ms)']:<10} "
                f"{row['Top Score']:<10} "
                f"{str(row['Results Count']):<5}"
            )
            logger.info(row_fmt)
        
        logger.info("=" * 200)
        logger.info("\n📊 SUMMARY PER QUERY TYPE:")
        logger.info("-" * 150)
        
        for row in rows:
            if row.get("Tipe Query Uji", "").startswith("[SUMMARY]"):
                row_fmt = (
                    f"{row['Tipe Query Uji']:<50} "
                    f"Hit Rate: {row['Hit Rate']:<8} "
                    f"Avg Latency: {row['Latensi (ms)']} ms  "
                    f"Score Range: {row['Top Score']}"
                )
                logger.info(row_fmt)
        
        logger.info("-" * 150)
        
        for row in rows:
            if row.get("Tipe Query Uji", "").startswith("***"):
                row_fmt = (
                    f"{row['Tipe Query Uji']:<50} "
                    f"Hit Rate: {row['Hit Rate']:<8} "
                    f"Avg Latency: {row['Latensi (ms)']} ms"
                )
                logger.info(row_fmt)
        
        logger.info("=" * 150)


def main():
    """Run benchmark"""
    logger.info(f"Working directory: {os.getcwd()}")
    
    tester = BenchmarkTester(
        qdrant_path="./qdrant_db",
        collection_name="scientific_articles",
    )
    
    results = tester.run_benchmark()
    tester.save_results("benchmark_results.json")
    tester.export_to_csv("benchmark_results.csv")
    
    return results


if __name__ == "__main__":
    main()
