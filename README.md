
# CrediLock — KYC Identity Verification System

A production-capable identity verification system built for secure, fraud-resistant onboarding.

## What it does

CrediLock verifies identity documents (Aadhaar, PAN) through a multi-step pipeline:

1. **Document Upload** — accepts images and PDFs
2. **Multi-engine OCR** — extracts text using PaddleOCR, EasyOCR, and Tesseract as fallbacks
3. **Document Detection** — identifies Aadhaar and PAN cards, extracts fields (UID, name, DOB, gender)
4. **Face Biometric Matching** — enrolls and matches face embeddings using face_recognition or FaceNet
5. **Risk Scoring** — assigns Low / Medium / High risk based on document integrity checks
6. **Cryptographic Anchoring** — signs verified documents using Ed25519 digital signatures for tamper-evident audit trails
7. **vCard Generation** — generates a verifiable identity card with transaction ID
## Prerequisites
- CMake installed
- Visual C++ Build Tools (Windows) or gcc (Linux)
- Python 3.10+

Note: Some dependencies (PaddleOCR, face_recognition) require system-level libraries. 
See requirements.txt for full list.

## Tech Stack

- **Backend:** Python, Flask
- **Database:** MongoDB
- **OCR:** PaddleOCR, EasyOCR, Tesseract
- **Biometrics:** face_recognition, FaceNet (facenet-pytorch)
- **Cryptography:** Ed25519 (via cryptography library)
- **Containerization:** Docker

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```



## API Endpoints

- `POST /upload` — upload and verify a document
- `POST /vcard` — generate anchored vCard with txid
- `GET /verify_anchor?txid=` — verify cryptographic signature
- `POST /match_biometric` — match face against enrolled documents
- `GET /flags` — get fraud flags for a file

