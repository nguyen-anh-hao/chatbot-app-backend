#!/usr/bin/env python3
"""
RAG Recipe Model for Production Use
Đóng gói mô hình RAG để sử dụng trong production với tốc độ cao
"""

import pickle
import joblib
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Optional

class RAGRecipeModel:
    """
    Mô hình RAG đã được tối ưu cho production
    
    Usage:
        # Lần đầu tiên: tạo và lưu mô hình
        model = RAGRecipeModel()
        model.create_and_save_model(df, embeddings)
        
        # Trong production: load và sử dụng
        model = RAGRecipeModel()
        model.load_model()
        results = model.search("chicken rice")
    """
    
    def __init__(self, model_dir: str = "./model_artifacts"):
        self.model_dir = Path(model_dir)
        self.sentence_model = None
        self.faiss_index = None
        self.recipes_df = None
        self.embeddings = None
        self.metadata = None
        
    def create_and_save_model(self, df: pd.DataFrame, embeddings: np.ndarray, 
                             model_name: str = "all-MiniLM-L6-v2"):
        """
        Tạo và lưu mô hình từ dữ liệu gốc
        
        Args:
            df: DataFrame chứa dữ liệu recipes
            embeddings: Numpy array chứa embeddings
            model_name: Tên của sentence transformer model
        """
        # Tạo thư mục
        self.model_dir.mkdir(exist_ok=True)
        
        # Lưu DataFrame
        df.to_pickle(self.model_dir / "recipes_df.pkl")
        
        # Lưu embeddings
        np.save(self.model_dir / "embeddings.npy", embeddings)
        
        # Tạo và lưu FAISS index
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.model_dir / "recipes_faiss.index"))
        
        # Lưu metadata
        metadata = {
            "model_name": model_name,
            "dimension": dimension,
            "num_recipes": len(df),
            "created_at": str(pd.Timestamp.now()),
            "version": "1.0"
        }
        
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Model saved to {self.model_dir}")
        print(f"📊 {len(df)} recipes, {dimension}D embeddings")
        
    def load_model(self, lazy_load: bool = True) -> Dict:
        """
        Load mô hình đã lưu
        
        Args:
            lazy_load: Nếu True, chỉ load khi cần thiết (tiết kiệm memory)
            
        Returns:
            Dictionary chứa metadata của mô hình
        """
        if not self.model_dir.exists():
            raise FileNotFoundError(f"❌ Model directory {self.model_dir} not found")
            
        # Load metadata
        with open(self.model_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        if not lazy_load:
            self._load_all_components()
        else:
            print("🚀 Model metadata loaded. Components will be loaded on demand.")
            
        print(f"✅ Model ready! {self.metadata['num_recipes']} recipes available.")
        return self.metadata
        
    def _load_all_components(self):
        """Load tất cả components của mô hình"""
        if self.sentence_model is None:
            print("📥 Loading sentence transformer...")
            self.sentence_model = SentenceTransformer(self.metadata["model_name"])
            
        if self.recipes_df is None:
            print("📥 Loading recipes dataframe...")
            self.recipes_df = pd.read_pickle(self.model_dir / "recipes_df.pkl")
            
        if self.embeddings is None:
            print("📥 Loading embeddings...")
            self.embeddings = np.load(self.model_dir / "embeddings.npy")
            
        if self.faiss_index is None:
            print("📥 Loading FAISS index...")
            self.faiss_index = faiss.read_index(str(self.model_dir / "recipes_faiss.index"))
            
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tìm kiếm recipes dựa trên query
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            List các dictionary chứa thông tin recipe
        """
        # Lazy loading
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer(self.metadata["model_name"])
        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(str(self.model_dir / "recipes_faiss.index"))
        if self.recipes_df is None:
            self.recipes_df = pd.read_pickle(self.model_dir / "recipes_df.pkl")
            
        # Encode query
        start_time = time.time()
        query_vector = self.sentence_model.encode([query])
        encode_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        D, I = self.faiss_index.search(np.array(query_vector), top_k)
        search_time = time.time() - start_time
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            recipe_info = {
                'rank': i + 1,
                'similarity_score': 1 / (1 + distance),
                'title': self.recipes_df.iloc[idx]['Title'],
                'ingredients': self.recipes_df.iloc[idx]['Cleaned_Ingredients'],
                'instructions': self.recipes_df.iloc[idx]['Instructions'],
                'image_name': self.recipes_df.iloc[idx]['Image_Name'],
                'encode_time': encode_time,
                'search_time': search_time
            }
            results.append(recipe_info)
            
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Tìm kiếm nhiều queries cùng lúc (tối ưu hơn)
        
        Args:
            queries: List các câu truy vấn
            top_k: Số lượng kết quả cho mỗi query
            
        Returns:
            List of lists chứa kết quả cho từng query
        """
        # Lazy loading
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer(self.metadata["model_name"])
        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(str(self.model_dir / "recipes_faiss.index"))
        if self.recipes_df is None:
            self.recipes_df = pd.read_pickle(self.model_dir / "recipes_df.pkl")
            
        # Encode tất cả queries cùng lúc
        query_vectors = self.sentence_model.encode(queries)
        
        all_results = []
        for i, query_vector in enumerate(query_vectors):
            D, I = self.faiss_index.search(np.array([query_vector]), top_k)
            
            results = []
            for j, (distance, idx) in enumerate(zip(D[0], I[0])):
                recipe_info = {
                    'rank': j + 1,
                    'similarity_score': 1 / (1 + distance),
                    'title': self.recipes_df.iloc[idx]['Title'],
                    'ingredients': self.recipes_df.iloc[idx]['Cleaned_Ingredients'],
                    'instructions': self.recipes_df.iloc[idx]['Instructions'],
                    'image_name': self.recipes_df.iloc[idx]['Image_Name']
                }
                results.append(recipe_info)
            all_results.append(results)
            
        return all_results
    
    def get_recipe_by_name(self, recipe_name: str, top_k: int = 5) -> List[Dict]:
        """
        Tìm kiếm theo tên món ăn
        
        Args:
            recipe_name: Tên món ăn
            top_k: Số lượng kết quả
            
        Returns:
            List các dictionary chứa thông tin recipe
        """
        if self.recipes_df is None:
            self.recipes_df = pd.read_pickle(self.model_dir / "recipes_df.pkl")
            
        matching_recipes = self.recipes_df[
            self.recipes_df['Title'].str.contains(recipe_name, case=False, na=False)
        ]
        
        if matching_recipes.empty:
            return []
            
        results = []
        for idx, recipe in matching_recipes.head(top_k).iterrows():
            recipe_info = {
                'title': recipe['Title'],
                'ingredients': recipe['Cleaned_Ingredients'],
                'instructions': recipe['Instructions'],
                'image_name': recipe['Image_Name']
            }
            results.append(recipe_info)
            
        return results
    
    def get_stats(self) -> Dict:
        """
        Lấy thống kê về mô hình
        
        Returns:
            Dictionary chứa thống kê
        """
        if self.recipes_df is None:
            self.recipes_df = pd.read_pickle(self.model_dir / "recipes_df.pkl")
            
        stats = {
            "total_recipes": len(self.recipes_df),
            "model_size_mb": sum(f.stat().st_size for f in self.model_dir.glob("*")) / (1024 * 1024),
            "created_at": self.metadata["created_at"] if self.metadata else "Unknown",
            "version": self.metadata["version"] if self.metadata else "Unknown"
        }
        return stats

# Utility functions cho production
def benchmark_search(model: RAGRecipeModel, queries: List[str], iterations: int = 10):
    """
    Benchmark tốc độ search
    """
    print("🔥 Benchmarking search performance...")
    
    total_times = []
    for i in range(iterations):
        start_time = time.time()
        for query in queries:
            model.search(query, top_k=5)
        total_time = time.time() - start_time
        total_times.append(total_time)
        
    avg_time = np.mean(total_times)
    avg_per_query = avg_time / len(queries)
    
    print(f"📊 Average time for {len(queries)} queries: {avg_time:.3f}s")
    print(f"📊 Average time per query: {avg_per_query:.3f}s")
    print(f"📊 Queries per second: {1/avg_per_query:.1f}")
    
    return {
        "avg_total_time": avg_time,
        "avg_per_query": avg_per_query,
        "queries_per_second": 1/avg_per_query
    }

if __name__ == "__main__":
    # Demo usage
    print("🚀 RAG Recipe Model Demo")
    
    # Load model
    model = RAGRecipeModel()
    
    try:
        metadata = model.load_model()
        print(f"📈 Model Stats: {model.get_stats()}")
        
        # Test search
        results = model.search("chicken and rice", top_k=3)
        print(f"\n🔍 Search results for 'chicken and rice':")
        for result in results:
            print(f"  {result['rank']}. {result['title']} (Score: {result['similarity_score']:.3f})")
            
        # Benchmark
        test_queries = ["chicken rice", "pasta", "dessert", "vegetarian", "spicy"]
        benchmark_search(model, test_queries, iterations=5)
        
    except FileNotFoundError:
        print("❌ Model not found. Please run the notebook to create the model first.")
