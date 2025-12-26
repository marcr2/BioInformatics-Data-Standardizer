"""
Vector Store for BIDS

ChromaDB wrapper for storing and retrieving:
- Error fingerprint vectors
- Associated fix scripts
- Success/failure metadata
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np

import chromadb
from chromadb.config import Settings

from .vectorizer import DataFingerprint


class ScriptStatus(Enum):
    """Status of a fix script."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"


@dataclass
class StoredScript:
    """A stored fix script with metadata."""
    id: str
    script_content: str
    status: ScriptStatus
    error_type: str
    source_file: str
    created_at: str
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    fingerprint_tokens: Optional[List[str]] = None


@dataclass
class QueryResult:
    """Result from a vector similarity query."""
    id: str
    distance: float
    script: StoredScript
    metadata: Dict[str, Any]


class VectorStore:
    """
    ChromaDB-based vector store for BIDS.
    
    Stores:
    - Error fingerprint vectors (TF-IDF)
    - Fix script content
    - Success/failure status
    - Metadata for retrieval and learning
    """
    
    COLLECTION_NAME = "bids_fingerprints"
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_dimension: int = 1000
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory for persistent storage (None for in-memory)
            embedding_dimension: Dimension of the TF-IDF vectors
        """
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        
        # Initialize ChromaDB client
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        # ChromaDB doesn't support custom embeddings directly, so we'll store
        # our TF-IDF vectors as metadata and use a simple embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "BIDS error fingerprints and fix scripts"}
        )
    
    def add_entry(
        self,
        fingerprint: DataFingerprint,
        script_content: str,
        status: ScriptStatus,
        error_type: str,
        source_file: str,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Add a new entry to the vector store.
        
        Args:
            fingerprint: DataFingerprint with vector
            script_content: Python fix script
            status: SUCCESS or FAILURE
            error_type: Type of error being fixed
            source_file: Source file name
            execution_time: Script execution time in seconds
            error_message: Error message if failed
            
        Returns:
            ID of the stored entry
        """
        entry_id = str(uuid.uuid4())
        
        # Prepare embedding (ensure it's the right shape)
        if fingerprint.vector is not None:
            embedding = fingerprint.vector.tolist()
            # Pad or truncate to expected dimension
            if len(embedding) < self.embedding_dimension:
                embedding = embedding + [0.0] * (self.embedding_dimension - len(embedding))
            else:
                embedding = embedding[:self.embedding_dimension]
        else:
            embedding = [0.0] * self.embedding_dimension
        
        # Prepare metadata
        metadata = {
            "status": status.value,
            "error_type": error_type,
            "source_file": source_file,
            "created_at": datetime.utcnow().isoformat(),
            "execution_time": execution_time or 0.0,
            "error_message": error_message or "",
            "token_count": len(fingerprint.tokens),
            "fingerprint_tokens": " ".join(fingerprint.tokens[:100])  # Store first 100 tokens
        }
        
        # Store in ChromaDB
        self.collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[script_content],
            metadatas=[metadata]
        )
        
        return entry_id
    
    def update_status(
        self,
        entry_id: str,
        status: ScriptStatus,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the status of an existing entry.
        
        Args:
            entry_id: ID of the entry to update
            status: New status
            execution_time: Execution time if applicable
            error_message: Error message if failed
            
        Returns:
            True if successful
        """
        try:
            # Get existing entry
            result = self.collection.get(ids=[entry_id], include=["metadatas"])
            
            if not result["ids"]:
                return False
            
            # Update metadata
            metadata = result["metadatas"][0]
            metadata["status"] = status.value
            if execution_time is not None:
                metadata["execution_time"] = execution_time
            if error_message is not None:
                metadata["error_message"] = error_message
            
            self.collection.update(
                ids=[entry_id],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception:
            return False
    
    def query_similar(
        self,
        fingerprint: DataFingerprint,
        top_k: int = 5,
        status_filter: Optional[ScriptStatus] = None
    ) -> List[QueryResult]:
        """
        Query for similar fingerprints.
        
        Args:
            fingerprint: Query fingerprint with vector
            top_k: Number of results to return
            status_filter: Filter by status (SUCCESS, FAILURE, or None for all)
            
        Returns:
            List of QueryResult objects
        """
        if fingerprint.vector is None:
            return []
        
        # Prepare query embedding
        embedding = fingerprint.vector.tolist()
        if len(embedding) < self.embedding_dimension:
            embedding = embedding + [0.0] * (self.embedding_dimension - len(embedding))
        else:
            embedding = embedding[:self.embedding_dimension]
        
        # Build where clause for filtering
        where = None
        if status_filter:
            where = {"status": status_filter.value}
        
        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            return []
        
        # Convert to QueryResult objects
        query_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, entry_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0.0
                
                script = StoredScript(
                    id=entry_id,
                    script_content=document,
                    status=ScriptStatus(metadata.get("status", "PENDING")),
                    error_type=metadata.get("error_type", ""),
                    source_file=metadata.get("source_file", ""),
                    created_at=metadata.get("created_at", ""),
                    execution_time=metadata.get("execution_time"),
                    error_message=metadata.get("error_message"),
                    fingerprint_tokens=metadata.get("fingerprint_tokens", "").split()
                )
                
                query_results.append(QueryResult(
                    id=entry_id,
                    distance=distance,
                    script=script,
                    metadata=metadata
                ))
        
        return query_results
    
    def get_successful_scripts(
        self,
        fingerprint: DataFingerprint,
        top_k: int = 3
    ) -> List[str]:
        """
        Get successful fix scripts similar to the given fingerprint.
        
        Args:
            fingerprint: Query fingerprint
            top_k: Number of scripts to return
            
        Returns:
            List of script content strings
        """
        results = self.query_similar(fingerprint, top_k, ScriptStatus.SUCCESS)
        return [r.script.script_content for r in results]
    
    def get_failed_scripts(
        self,
        fingerprint: DataFingerprint,
        top_k: int = 3
    ) -> List[str]:
        """
        Get failed fix scripts similar to the given fingerprint.
        
        Args:
            fingerprint: Query fingerprint
            top_k: Number of scripts to return
            
        Returns:
            List of script content strings
        """
        results = self.query_similar(fingerprint, top_k, ScriptStatus.FAILURE)
        return [r.script.script_content for r in results]
    
    def get_entry(self, entry_id: str) -> Optional[StoredScript]:
        """
        Get a specific entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            StoredScript or None if not found
        """
        try:
            result = self.collection.get(
                ids=[entry_id],
                include=["documents", "metadatas"]
            )
            
            if not result["ids"]:
                return None
            
            metadata = result["metadatas"][0]
            document = result["documents"][0]
            
            return StoredScript(
                id=entry_id,
                script_content=document,
                status=ScriptStatus(metadata.get("status", "PENDING")),
                error_type=metadata.get("error_type", ""),
                source_file=metadata.get("source_file", ""),
                created_at=metadata.get("created_at", ""),
                execution_time=metadata.get("execution_time"),
                error_message=metadata.get("error_message"),
                fingerprint_tokens=metadata.get("fingerprint_tokens", "").split()
            )
            
        except Exception:
            return None
    
    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict with statistics
        """
        try:
            count = self.collection.count()
            
            # Get status distribution
            all_results = self.collection.get(include=["metadatas"])
            
            status_counts = {"SUCCESS": 0, "FAILURE": 0, "PENDING": 0}
            error_types = {}
            
            for metadata in all_results.get("metadatas", []):
                status = metadata.get("status", "PENDING")
                status_counts[status] = status_counts.get(status, 0) + 1
                
                error_type = metadata.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            return {
                "total_entries": count,
                "status_distribution": status_counts,
                "error_types": error_types,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def clear(self) -> bool:
        """
        Clear all entries from the store.
        
        Returns:
            True if successful
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "BIDS error fingerprints and fix scripts"}
            )
            return True
        except Exception:
            return False
    
    def export_to_json(self, file_path: str) -> bool:
        """
        Export all entries to a JSON file.
        
        Args:
            file_path: Path to output file
            
        Returns:
            True if successful
        """
        try:
            all_results = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            entries = []
            for i, entry_id in enumerate(all_results.get("ids", [])):
                entries.append({
                    "id": entry_id,
                    "script_content": all_results["documents"][i],
                    "metadata": all_results["metadatas"][i],
                    "embedding": all_results["embeddings"][i] if all_results.get("embeddings") else None
                })
            
            with open(file_path, 'w') as f:
                json.dump(entries, f, indent=2)
            
            return True
            
        except Exception:
            return False
    
    def import_from_json(self, file_path: str) -> int:
        """
        Import entries from a JSON file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            Number of entries imported
        """
        try:
            with open(file_path, 'r') as f:
                entries = json.load(f)
            
            count = 0
            for entry in entries:
                self.collection.add(
                    ids=[entry["id"]],
                    embeddings=[entry.get("embedding", [0.0] * self.embedding_dimension)],
                    documents=[entry["script_content"]],
                    metadatas=[entry["metadata"]]
                )
                count += 1
            
            return count
            
        except Exception:
            return 0


def get_vector_store(persist_directory: Optional[str] = None) -> VectorStore:
    """
    Factory function to get a VectorStore instance.
    
    Args:
        persist_directory: Directory for persistent storage
        
    Returns:
        VectorStore instance
    """
    return VectorStore(persist_directory)
