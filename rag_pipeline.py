# rag_pipeline.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict
import os
import config

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for allergen knowledge"""
    
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load and index allergen database if empty
        if self.collection.count() == 0:
            self._index_allergen_database()
    
    def _index_allergen_database(self):
        """Index allergen database into ChromaDB"""
        df = pd.read_csv("data/allergen_database.csv")
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create rich document text
            doc = f"""Ingredient: {row['ingredient_name']}
Common Name: {row['common_name']}
Category: {row['category']}
Allergen Group: {row['allergen_group']}
Risk Level: {row['risk_level']}
Description: {row['description']}
Source: {row['source']}"""
            
            documents.append(doc)
            metadatas.append({
                'ingredient_name': row['ingredient_name'],
                'allergen_group': row['allergen_group'],
                'risk_level': row['risk_level']
            })
            ids.append(f"ingredient_{idx}")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Indexed {len(documents)} ingredients into ChromaDB")
    
    def retrieve_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve similar ingredients from knowledge base"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        retrieved = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return retrieved
    
    def get_context_for_ingredient(self, ingredient: str) -> str:
        """Get relevant context for an ingredient"""
        results = self.retrieve_similar(ingredient, n_results=3)
        
        if not results:
            return "No specific information found in knowledge base."
        
        context = "Retrieved from knowledge base:\n\n"
        for i, result in enumerate(results):
            context += f"Source {i+1}:\n{result['document']}\n\n"
        
        return context