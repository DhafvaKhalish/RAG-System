# RAG System with Qdrant and Huggingface

Sistem ini menerapkan Retrieval-Augmented Generation (RAG) menggunakan Python, Qdrant, dan model LLM dari Huggingface.

## Fitur
- Penyematan dokumen dengan SentenceTransformer
- Penyimpanan embedding di Qdrant
- FastAPI untuk menerima pertanyaan pengguna
- Prompting ke LLM berbasis konteks dokumen

## Cara Menjalankan di Google Colab
1. Jalankan `main.ipynb`
2. Gunakan API di `/ask` dengan metode POST dan body JSON `{ "question": "..." }`

## Contoh Pertanyaan
```
{
  "question": "Apa itu Python?"
}
```

## Hasil
```
{
  "answer": "Python adalah bahasa pemrograman populer..."
}
```
