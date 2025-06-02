# IR-System-BE

Backend untuk Sistem Information Retrieval menggunakan FastAPI.

## Daftar Isi

-   [Persyaratan](#persyaratan)
-   [Cara Menjalankan](#cara-menjalankan)
    -   [Menggunakan Virtual Environment](#menggunakan-virtual-environment)
    -   [Menggunakan Docker](#menggunakan-docker)
-   [Endpoint API](#endpoint-api)
-   [Struktur Proyek](#struktur-proyek)
-   [Implementasi Algoritma](#implementasi-algoritma)

## Persyaratan

-   Python 3.8+
-   pip
-   Docker (opsional, untuk menjalankan dalam container)

## Cara Menjalankan

### Menggunakan Virtual Environment

1. Buat dan aktifkan virtual environment:

```bash
# Windows
python -m venv venv
source venv/Scripts/activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
python main.py
```

4. Akses aplikasi di [http://localhost:8080](http://localhost:8080)
5. Dokumentasi API tersedia di [http://localhost:8080/docs](http://localhost:8080/docs)

### Menggunakan Docker

#### Metode 1: Docker Run

1. Build Docker image:

```bash
docker build -t ir-system-be .
```

2. Jalankan container:

```bash
docker run -p 8080:8080 ir-system-be
```

#### Metode 2: Docker Compose (Direkomendasikan)

1. Jalankan dengan Docker Compose:

```bash
docker-compose up
```

2. Untuk menjalankan di background:

```bash
docker-compose up -d
```

3. Untuk menghentikan layanan:

```bash
docker-compose down
```

4. Akses aplikasi di [http://localhost:8080](http://localhost:8080)
5. Dokumentasi API tersedia di [http://localhost:8080/docs](http://localhost:8080/docs)

## Endpoint API

### Endpoint Dasar

-   `GET /`: Halaman utama
-   `GET /health`: Endpoint kesehatan untuk memeriksa status layanan

### Endpoint Dokumen

-   `POST /documents/upload`: Upload dokumen
-   `POST /documents/parse`: Parse dokumen dari direktori

### Endpoint Retrieval

-   `POST /retrieval/query/interactive`: Query interaktif
-   `POST /retrieval/query/batch`: Batch query
-   `GET /retrieval/inverted-file`: Mendapatkan inverted file

## Struktur Proyek

```
IR-System-BE/
├── app/                        # Package utama aplikasi
│   ├── core/                   # Konfigurasi dan fungsi inti
│   │   └── config.py           # Konfigurasi aplikasi
│   ├── data/                   # Data seperti stopwords, model Word2Vec, dll.
│   ├── models/                 # Model/schema Pydantic
│   │   └── query_models.py     # Model untuk query input dan output
│   ├── routers/                # Router FastAPI
│   │   ├── documents.py        # Router untuk endpoint dokumen
│   │   └── retrieval.py        # Router untuk endpoint retrieval
│   ├── services/               # Layanan bisnis
│   │   ├── document_service.py # Service untuk mengelola dokumen
│   │   ├── retrieval_service.py# Service untuk retrieval dan TF-IDF
│   │   └── query_expansion_service.py # Service untuk query expansion
│   └── utils/                  # Utilitas
│       ├── text_preprocessing.py # Fungsi preprocessing teks
│       └── evaluation.py       # Fungsi evaluasi (MAP, dll.)
├── main.py                     # Entry point aplikasi
├── Dockerfile                  # Konfigurasi Docker
├── docker-compose.yml          # Konfigurasi Docker Compose
└── requirements.txt            # Dependensi Python
```

## Implementasi Algoritma

### 1. Preprocessing

-   Stemming untuk Bahasa Indonesia
-   Stopword removal

### 2. Inverted File & TF-IDF

-   Pembuatan inverted file
-   Implementasi berbagai skema TF (raw, log, binary, augmented)
-   Pembobotan IDF
-   Normalisasi vektor

### 3. Query Expansion dengan Word2Vec

-   Training model Word2Vec dari corpus
-   Ekspansi query berdasarkan similaritas term
-   Adjustable threshold untuk query expansion

### 4. Evaluasi

-   Mean Average Precision (MAP)
-   Perbandingan hasil sebelum dan sesudah query expansion
