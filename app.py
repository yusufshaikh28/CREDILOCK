import os
import io
import re
import json
import base64
import hashlib
import logging
import datetime as dt
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import imagehash

# optional heavy deps
try:
    import cv2
except Exception:
    cv2 = None

# ML helpers
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except Exception:
    IsolationForest = None
    StandardScaler = None

# OCR libs (optional)
PADDLE_AVAILABLE = False
EASY_AVAILABLE = False
TESSERACT_AVAILABLE = False
_paddle = None
_easy = None
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# face libs (optional)
FACE_REC_AVAILABLE = False
FACENET_AVAILABLE = False
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except Exception:
    FACE_REC_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    FACENET_AVAILABLE = True
except Exception:
    FACENET_AVAILABLE = False

# pdf
try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# qr
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False

# cryptography for Ed25519
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

# ---------- config ----------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
KEYS_FOLDER = os.path.join(BASE_DIR, 'keys')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KEYS_FOLDER, exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'pdf', 'csv', 'json'}
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'credilock')

app = Flask(__name__, static_folder='ui', template_folder='ui')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------- db ----------
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
raw_col = db['raw_files']
flags_col = db['file_flags']
ml_col = db['file_ml']
profiles_col = db['biometric_profiles']
verified_col = db['verified_records']
ledger_col = db['blockchain_ledger']

try:
    raw_col.create_index([('upload_ts', ASCENDING)])
    flags_col.create_index([('file_id', ASCENDING)])
    profiles_col.create_index([('profile_id', ASCENDING)], unique=True)
    ledger_col.create_index([('txid', ASCENDING)], unique=True)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credilock")

# ---------- helpers ----------
def serialize_for_mongo(x):
    if isinstance(x, dict):
        return {k: serialize_for_mongo(v) for k, v in x.items()}
    if isinstance(x, list):
        return [serialize_for_mongo(i) for i in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (bytes, bytearray)):
        return x.decode('latin1', errors='ignore')
    if isinstance(x, dt.datetime):
        return x.isoformat()
    return x

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def safe_preview(text, n=4000):
    if not text:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='ignore')
        except Exception:
            text = str(text)
    try:
        text = text.replace('\\n', '\n')
    except Exception:
        pass
    return text[:n]

def clean_newlines_only(text):
    if not text:
        return ""
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r'\n{3,}', '\n\n', t)
    return "\n".join([ln.rstrip() for ln in t.splitlines()]).strip()

# ---------- OCR ----------
_paddle = None

def ensure_paddle():
    global _paddle
    if not PADDLE_AVAILABLE:
        return
    if _paddle is not None:
        return
    try:
        import paddle
        from paddleocr import PaddleOCR
        use_gpu = False
        try:
            use_gpu = bool(paddle.is_compiled_with_cuda())
        except Exception:
            use_gpu = False
        _paddle = PaddleOCR(lang="en", use_textline_orientation=True, use_gpu=use_gpu, show_log=False)
        logger.info("PaddleOCR ready (GPU=%s)", use_gpu)
    except Exception as e:
        logger.warning("PaddleOCR disabled: %s", e)
        _paddle = None

def ensure_easy():
    global _easy
    if not EASY_AVAILABLE:
        return
    if _easy is not None:
        return
    try:
        _easy = easyocr.Reader(['en', 'hi'], gpu=False, verbose=False)
    except Exception:
        try:
            _easy = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception:
            _easy = None

def preprocess_variants_for_ocr(img_bytes):
    out = []
    try:
        base = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return out
    out.append(('orig', base))
    try:
        a = ImageOps.autocontrast(base, cutoff=1)
        a = ImageEnhance.Sharpness(a).enhance(1.2)
        out.append(('autocontrast', a))
    except Exception:
        pass
    try:
        w, h = base.size
        if w < 1200:
            scale = max(1.5, 1200.0 / max(1, w))
            resized = base.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            out.append(('resized', resized))
    except Exception:
        pass
    try:
        inv = ImageOps.invert(base)
        out.append(('invert', inv))
    except Exception:
        pass
    try:
        if cv2 is not None:
            arr = np.array(base)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            out.append(('clahe', Image.fromarray(final)))
    except Exception:
        pass
    return out

def tesseract_ocr(pil_img, lang=None):
    try:
        import pytesseract
        cfg = ''
        if lang:
            cfg = f'-l {lang}'
        return pytesseract.image_to_string(pil_img, config=cfg) or ""
    except Exception:
        return ""

_devanagari_re = re.compile(r'[\u0900-\u097F]')

def is_devanagari_present(text):
    return bool(_devanagari_re.search(text or ""))

def filter_garbage_lines(raw_text, preserve_devanagari=True):
    if not raw_text:
        return ""
    lines = [ln for ln in raw_text.splitlines()]
    out_lines = []
    for ln in lines:
        s = ln.strip()
        if s == "":
            if len(out_lines) == 0 or out_lines[-1].strip() != "":
                out_lines.append("")
            continue
        if preserve_devanagari and _devanagari_re.search(s):
            if len(re.findall(_devanagari_re, s)) >= 2:
                out_lines.append(s)
            else:
                clean = re.sub(r'[^\u0900-\u097F\s\d\.\,\-]', '', s).strip()
                if len(re.findall(_devanagari_re, clean)) >= 2:
                    out_lines.append(clean)
            continue
        non_symbol_count = len(re.findall(r'[A-Za-z0-9]', s))
        total = max(1, len(s))
        symbol_ratio = 1.0 - (non_symbol_count / total)
        if symbol_ratio > 0.45:
            continue
        if re.match(r'^(.)(\1+)$', s):
            continue
        if len(s) < 3:
            if re.match(r'^\d{3,}$', s):
                out_lines.append(s)
            else:
                continue
        tokens = s.split()
        single_token_frac = sum(1 for t in tokens if len(t) == 1) / max(1, len(tokens))
        if single_token_frac > 0.5 and len(tokens) >= 3:
            continue
        letters = len(re.findall(r'[A-Za-z]', s))
        if letters / total < 0.25 and not re.search(r'\d{3,}', s):
            continue
        out_lines.append(s)
    cleaned = []
    prev_blank = False
    for ln in out_lines:
        if ln.strip() == "":
            if not prev_blank:
                cleaned.append("")
                prev_blank = True
        else:
            cleaned.append(ln)
            prev_blank = False
    return "\n".join(cleaned).strip()

def ocr_image(img_bytes):
    ensure_paddle()
    ensure_easy()
    variants = preprocess_variants_for_ocr(img_bytes)
    for name, pil_img in variants:
        arr = np.array(pil_img.convert('RGB'))
        if PADDLE_AVAILABLE and _paddle is not None:
            try:
                raw = _paddle.ocr(arr, cls=True)
                lines = []
                for r in raw:
                    try:
                        lines.append(r[1][0])
                    except Exception:
                        pass
                text = "\n".join(lines)
                if text and text.strip():
                    return clean_newlines_only(text), [], 'paddle'
            except Exception:
                pass
        if EASY_AVAILABLE and _easy is not None:
            try:
                out = _easy.readtext(arr, detail=1, paragraph=False)
                lines = [t for bbox, t, conf in out]
                text = "\n".join(lines)
                if text and text.strip():
                    return clean_newlines_only(text), [], 'easy'
            except Exception:
                pass
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return "", [], "none"
    arr = np.array(img)
    if PADDLE_AVAILABLE and _paddle is not None:
        try:
            raw = _paddle.ocr(arr, cls=True)
            lines = [r[1][0] for r in raw if len(r) >= 2]
            text = clean_newlines_only("\n".join(lines))
            if text:
                return text, [], 'paddle'
        except Exception:
            pass
    if EASY_AVAILABLE and _easy is not None:
        try:
            out = _easy.readtext(arr, detail=1, paragraph=False)
            lines = [t for bbox, t, conf in out]
            text = clean_newlines_only("\n".join(lines))
            if text:
                return text, [], 'easy'
        except Exception:
            pass
    if TESSERACT_AVAILABLE:
        try:
            txt = tesseract_ocr(img, lang='hin')
            txt = clean_newlines_only(txt)
            if txt and len(txt) > 1:
                return txt, [], 'tesseract-hin'
        except Exception:
            pass
        try:
            txt = tesseract_ocr(img, lang='eng')
            txt = clean_newlines_only(txt)
            if txt and len(txt) > 1:
                return txt, [], 'tesseract-en'
        except Exception:
            pass
    return "", [], "none"

def micro_qr_scan(img_bytes):
    results = []
    if not PYZBAR_AVAILABLE:
        return results
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        arr = np.array(pil)
        dec = pyzbar_decode(arr)
        for d in dec:
            try:
                txt = d.data.decode('utf-8', errors='ignore')
            except Exception:
                txt = str(d.data)
            results.append({'data': txt, 'type': d.type, 'rect': [d.rect.left, d.rect.top, d.rect.width, d.rect.height]})
    except Exception:
        pass
    return results

def compute_phash_for_image(path_or_bytes):
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            img = Image.open(io.BytesIO(path_or_bytes)).convert('RGB')
        else:
            img = Image.open(path_or_bytes).convert('RGB')
        ph = imagehash.phash(img)
        return str(ph)
    except Exception:
        return None

def phash_distance(h1, h2):
    try:
        if h1 is None or h2 is None:
            return None
        return imagehash.hex_to_hash(str(h1)) - imagehash.hex_to_hash(str(h2))
    except Exception:
        return None

# ---------- face ----------
_facenet_mtcnn = None
_facenet_model = None

def ensure_facenet():
    global _facenet_mtcnn, _facenet_model
    if not FACENET_AVAILABLE:
        return
    if _facenet_mtcnn is not None and _facenet_model is not None:
        return
    try:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        _facenet_mtcnn = MTCNN(keep_all=False, device=dev)
        _facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
    except Exception:
        _facenet_mtcnn = None
        _facenet_model = None

def enroll_face_facenet(img_bytes):
    ensure_facenet()
    if _facenet_mtcnn is None or _facenet_model is None:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        boxes, probs = _facenet_mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        valid = []
        for box, prob in zip(boxes, probs):
            if prob is None or float(prob) < 0.90:
                continue
            x1, y1, x2, y2 = box
            fw, fh = x2 - x1, y2 - y1
            if fw < 40 or fh < 40:
                continue
            if fw > w * 0.95 or fh > h * 0.95:
                continue
            valid.append((box, prob))
        if not valid:
            return None
        face = _facenet_mtcnn(img)
        if face is None:
            return None
        with torch.no_grad():
            if face.dim() == 3:
                face = face.unsqueeze(0)
            emb = _facenet_model(face)
            return emb[0].cpu().numpy().astype(float).tolist()
    except Exception:
        return None

def enroll_best_embedding_from_bytes(img_bytes):
    if FACENET_AVAILABLE:
        try:
            emb = enroll_face_facenet(img_bytes)
            if emb:
                return {'backend': 'facenet', 'embedding': emb}
        except Exception:
            pass
    return None

def normalize_vec(v):
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a) + 1e-9
    return (a / n).tolist()

def match_embeddings(embA, embB, backend_hint=None):
    if embA is None or embB is None:
        return 0.0
    try:
        a = np.array(embA, dtype=float)
        b = np.array(embB, dtype=float)
        if a.shape != b.shape:
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    except Exception:
        return 0.0

def face_confidence_from_similarity(sim):
    """
    Map cosine similarity → 0-100 confidence.
    FaceNet cosine sim: ~0.2 = different people, ~1.0 = same person.
    Returns None if sim is None — never return a fake score.
    """
    if sim is None:
        return None
    try:
        s = float(sim)
    except Exception:
        return None
    low = 0.20
    high = 1.00
    s = max(low, min(high, s))
    frac = (s - low) / (high - low)
    return int(round(frac * 100))

# ---------- ML metrics ----------
def compute_and_persist_ml_metrics_for_file(raw_doc_id, file_doc):
    ml_metrics = {
        'note': 'not_computed',
        'pool_count': 0,
        'nearest_sim': None,
        'nearest_file_id': None,
        'nearest_l2': None,
        'iso_score': None,
        'is_outlier': None,
        'duplicate_exact': False,
        'confidence': None
    }
    try:
        emb = file_doc.get('face_embedding')
        if not emb:
            ml_metrics['note'] = 'no_face_embedding'
            ml_metrics['confidence'] = None
            try:
                ml_col.insert_one({'file_id': str(raw_doc_id), 'created_at': dt.datetime.utcnow(), 'ml_metrics': serialize_for_mongo(ml_metrics)})
            except Exception:
                pass
            try:
                raw_col.update_one({'_id': ObjectId(raw_doc_id)}, {'$set': {'ml_tag': serialize_for_mongo(ml_metrics)}})
            except Exception:
                pass
            return ml_metrics

        probe = np.array(emb, dtype=float)
        if probe.size == 0 or np.isnan(probe).any() or np.linalg.norm(probe) < 1e-12:
            ml_metrics['note'] = 'invalid_embedding'
            ml_metrics['confidence'] = None
            try:
                ml_col.insert_one({'file_id': str(raw_doc_id), 'created_at': dt.datetime.utcnow(), 'ml_metrics': serialize_for_mongo(ml_metrics)})
            except Exception:
                pass
            return ml_metrics

        pool_embs = []
        pool_ids = []
        try:
            # CRITICAL: exclude self from pool
            cursor = raw_col.find(
                {'face_embedding': {'$exists': True, '$ne': None}, '_id': {'$ne': ObjectId(raw_doc_id)}}
            ).sort('upload_ts', -1).limit(2000)
            for d in cursor:
                did = d.get('_id')
                e = d.get('face_embedding')
                if not e:
                    continue
                arr = np.array(e, dtype=float)
                if arr.shape != probe.shape:
                    continue
                pool_embs.append(arr)
                pool_ids.append(str(did))
        except Exception:
            pass

        pool_count = len(pool_embs)
        ml_metrics['pool_count'] = int(pool_count)

        if pool_count == 0:
            ml_metrics['note'] = 'empty_pool'
            ml_metrics['nearest_sim'] = None
            ml_metrics['confidence'] = None
            ml_metrics['is_outlier'] = False
            try:
                ml_col.insert_one({'file_id': str(raw_doc_id), 'created_at': dt.datetime.utcnow(), 'ml_metrics': serialize_for_mongo(ml_metrics)})
            except Exception:
                pass
            try:
                raw_col.update_one({'_id': ObjectId(raw_doc_id)}, {'$set': {'ml_tag': serialize_for_mongo(ml_metrics)}})
            except Exception:
                pass
            return ml_metrics

        try:
            X = np.stack(pool_embs, axis=0).astype(float)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            pn = probe / (np.linalg.norm(probe) + 1e-9)
            sims = (Xn @ pn).tolist()
            idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
            chosen_idx = None
            duplicate_flag = False
            for i in idxs:
                l2 = float(np.linalg.norm(pn - Xn[i]))
                if l2 <= 1e-8:
                    chosen_idx = i
                    duplicate_flag = True
                    break
                else:
                    chosen_idx = i
                    duplicate_flag = False
                    break
            if chosen_idx is not None:
                best_sim = float(sims[chosen_idx])
                best_id = pool_ids[chosen_idx]
                best_l2 = float(np.linalg.norm(pn - Xn[chosen_idx]))
                ml_metrics.update({
                    'nearest_sim': best_sim,
                    'nearest_file_id': best_id,
                    'nearest_l2': best_l2,
                    'duplicate_exact': bool(duplicate_flag)
                })
                ml_metrics['note'] = 'computed'
                ml_metrics['confidence'] = face_confidence_from_similarity(best_sim)
            else:
                ml_metrics['note'] = 'computed_no_candidate'
                ml_metrics['confidence'] = None
        except Exception:
            ml_metrics['note'] = 'nearest_failed'
            ml_metrics['confidence'] = None

        if pool_count >= 8 and IsolationForest is not None and StandardScaler is not None:
            try:
                scaler = StandardScaler()
                Xs_scaled = scaler.fit_transform(Xn)
                iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
                iso.fit(Xs_scaled)
                probe_scaled = scaler.transform(pn.reshape(1, -1))
                iso_score = float(iso.decision_function(probe_scaled)[0])
                iso_pred = int(iso.predict(probe_scaled)[0])
                ml_metrics['iso_score'] = iso_score
                ml_metrics['is_outlier'] = bool(iso_pred == -1)
            except Exception:
                ml_metrics['iso_score'] = None
                ml_metrics['is_outlier'] = None
        else:
            ml_metrics['is_outlier'] = False
            ml_metrics['iso_score'] = None

        try:
            ml_col.insert_one({'file_id': str(raw_doc_id), 'filename': file_doc.get('filename'), 'created_at': dt.datetime.utcnow(), 'ml_metrics': serialize_for_mongo(ml_metrics)})
        except Exception:
            pass
        try:
            raw_col.update_one({'_id': ObjectId(raw_doc_id)}, {'$set': {'ml_tag': serialize_for_mongo(ml_metrics)}})
        except Exception:
            pass

        return ml_metrics
    except Exception as e:
        logger.exception("ml compute failed: %s", e)
        ml_metrics = {'error': 'ml_failed', 'detail': str(e), 'confidence': None}
        try:
            ml_col.insert_one({'file_id': str(raw_doc_id), 'created_at': dt.datetime.utcnow(), 'ml_metrics': serialize_for_mongo(ml_metrics)})
        except Exception:
            pass
        return ml_metrics

def extract_text_from_pdf(path):
    if not PDF_AVAILABLE:
        return ""
    try:
        with open(path, 'rb') as fh:
            reader = PyPDF2.PdfReader(fh)
            text = ""
            for p in reader.pages:
                try:
                    pg = p.extract_text()
                    if pg:
                        text += pg + "\n"
                except Exception:
                    pass
            return clean_newlines_only(text)
    except Exception:
        return ""

# ---------- crypto ----------
def load_or_create_ed25519_keypair():
    private_path = os.path.join(KEYS_FOLDER, 'ed25519_private.pem')
    public_path = os.path.join(KEYS_FOLDER, 'ed25519_public.pem')
    if os.path.exists(private_path) and os.path.exists(public_path):
        try:
            with open(private_path, 'rb') as fh:
                priv = serialization.load_pem_private_key(fh.read(), password=None, backend=default_backend())
            with open(public_path, 'rb') as fh:
                pub_pem = fh.read()
                pub = serialization.load_pem_public_key(pub_pem, backend=default_backend())
            return priv, pub, pub_pem
        except Exception:
            logger.exception("Failed to load existing keys; will regenerate.")
    if not CRYPTO_AVAILABLE:
        logger.warning("Cryptography not available: cannot create signing keys.")
        return None, None, None
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_pem = priv.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    pub_pem = pub.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    try:
        with open(private_path, 'wb') as fh:
            fh.write(priv_pem)
        with open(public_path, 'wb') as fh:
            fh.write(pub_pem)
        os.chmod(private_path, 0o600)
    except Exception:
        logger.exception("Failed to write keys to disk; continuing with in-memory keys.")
    return priv, pub, pub_pem

if CRYPTO_AVAILABLE:
    _sign_priv, _sign_pub, _sign_pub_pem = load_or_create_ed25519_keypair()
    if _sign_priv is None or _sign_pub is None:
        logger.warning("Ed25519 key not available: anchor signing disabled.")
else:
    _sign_priv = None
    _sign_pub = None
    _sign_pub_pem = None

def pubkey_fingerprint(pub_pem_bytes):
    if not pub_pem_bytes:
        return None
    return hashlib.sha256(pub_pem_bytes).hexdigest()

def sign_txid_with_privkey(txid_hex):
    if _sign_priv is None:
        return None
    try:
        sig = _sign_priv.sign(txid_hex.encode('utf-8'))
        return base64.b64encode(sig).decode('ascii')
    except Exception:
        logger.exception("Signing failed")
        return None

def verify_signature_for_txid(pub_pem_bytes, txid_hex, signature_b64):
    try:
        pub = serialization.load_pem_public_key(pub_pem_bytes, backend=default_backend())
        sig = base64.b64decode(signature_b64)
        pub.verify(sig, txid_hex.encode('utf-8'))
        return True
    except Exception:
        return False

def blockchain_anchor_signed(payload):
    dump = json.dumps(payload, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8')
    ts = dt.datetime.utcnow().isoformat()
    txid = hashlib.sha256(dump + ts.encode('utf-8')).hexdigest()
    sig_b64 = None
    pk_fp = None
    pub_pem_bytes = None
    try:
        if _sign_priv is not None:
            sig_b64 = sign_txid_with_privkey(txid)
            pub_pem_bytes = _sign_pub_pem
            pk_fp = pubkey_fingerprint(pub_pem_bytes)
    except Exception:
        logger.exception("Signing attempt failed")
    tx = {'txid': txid, 'payload': payload, 'signature': sig_b64, 'pubkey_fingerprint': pk_fp, 'pubkey_pem': base64.b64encode(pub_pem_bytes).decode('ascii') if pub_pem_bytes else None, 'ts': dt.datetime.utcnow()}
    try:
        ledger_col.insert_one(serialize_for_mongo(tx))
    except Exception:
        logger.debug("ledger insert may have failed or duplicate (non-fatal)")
    return tx

def generate_vcard(name, email=None, phone=None, txid=None):
    lines = ['BEGIN:VCARD', 'VERSION:3.0', 'FN:' + (name or '')]
    if email:
        lines.append('EMAIL;TYPE=INTERNET:' + email)
    if phone:
        lines.append('TEL;TYPE=CELL:' + phone)
    if txid:
        lines.append('NOTE:Credilock TXID: ' + txid)
    lines.append('END:VCARD')
    return "\n".join(lines)

# ---------- ID detection ----------
_AADHAAR_RE = re.compile(r'\b([1-9]\d{11})\b')
_AADHAAR_RE_SPACED = re.compile(r'\b(\d{4}\s+\d{4}\s+\d{4})\b')
_PAN_RE = re.compile(r'\b([A-Z]{5}\d{4}[A-Z])\b', re.I)
_DOB_RE = re.compile(r'\b(0?[1-9]|[12][0-9]|3[01])[-/\.](0?[1-9]|1[0-2])[-/\.](\d{4}|\d{2})\b')
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
_GOVT_TOKENS = {'GOVERNMENT', 'INDIA', 'भारत', 'AADHAAR', 'UIDAI', 'UNIQUE', 'IDENTITY'}

def extract_name_near_line(lines, idx, window=4):
    n = len(lines)
    candidates = []
    for i in range(max(0, idx - window), min(n, idx + window + 1)):
        ln = lines[i].strip()
        if not ln:
            continue
        if any(tok in ln.upper() for tok in _GOVT_TOKENS):
            continue
        if re.search(r'\d', ln):
            continue
        if re.match(r'^(MALE|FEMALE|DOB|DATE|GENDER|NAME)$', ln.strip(), re.I):
            continue
        candidates.append((i, ln))
    for i, ln in candidates:
        if len(ln.split()) >= 2 and len(ln) <= 40:
            return ln
    if candidates:
        return max(candidates, key=lambda x: len(x[1]))[1]
    return None

def detect_pan_from_text(text):
    txt = (text or "").strip()
    if not txt:
        return None
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    pan_match = _PAN_RE.search(txt)
    if not pan_match:
        return {'is_pan_like': False, 'pan_fields': {}, 'notes': [], 'risk_score': 0}
    pan = pan_match.group(1).upper()
    pan_idx = None
    for i, ln in enumerate(lines):
        if pan in ln.upper() or pan.replace(' ', '') in ln.upper():
            pan_idx = i
            break
    name_guess = None
    dob_guess = None
    notes = []
    risk = 0
    if pan_idx is not None:
        name_guess = extract_name_near_line(lines, pan_idx, window=3)
    if not name_guess:
        for ln in lines[:8]:
            if re.search(r'[A-Za-z]', ln) and not re.search(r'\d', ln) and len(ln.split()) <= 5:
                if not any(tok in ln.upper() for tok in _GOVT_TOKENS):
                    name_guess = ln.strip()
                    break
    if pan_idx is not None:
        for i in range(max(0, pan_idx - 4), min(len(lines), pan_idx + 4)):
            m = _DOB_RE.search(lines[i])
            if m:
                dob_guess = m.group(0)
                break
        if not dob_guess:
            for i in range(max(0, pan_idx - 4), min(len(lines), pan_idx + 4)):
                m = _YEAR_RE.search(lines[i])
                if m:
                    dob_guess = m.group(0)
                    break
    if not dob_guess:
        m = _DOB_RE.search(txt)
        if m:
            dob_guess = m.group(0)
    if not name_guess:
        notes.append('name_not_found_near_pan')
        risk += 35
    if not dob_guess:
        notes.append('dob_not_found_near_pan')
        risk += 20
    if re.search(r'INCOME\s+TAX|PERMANENT\s+ACCOUNT|GOVERNMENT\s+OF\s+INDIA', txt, re.I):
        risk = max(0, risk - 15)
        notes.append('govt_token_found')
    risk = max(0, min(100, risk))
    return {'is_pan_like': True, 'pan_fields': {'pan_number': pan, 'name': name_guess, 'dob': dob_guess}, 'notes': notes, 'risk_score': int(risk)}

def detect_aadhaar_from_text(text, micro_qr_list):
    txt = (text or "").strip()
    if not txt:
        return None
    lower = txt.lower()
    aad_m = _AADHAAR_RE.search(txt) or _AADHAAR_RE_SPACED.search(txt)
    aad_raw = aad_m.group(1).replace(' ', '') if aad_m else None
    aad_masked = mask_aadhaar(aad_raw)
    is_like = False
    notes = []
    if 'aadhaar' in lower or 'aadhar' in lower or 'uidai' in lower:
        is_like = True
    aad_m2 = _AADHAAR_RE.search(txt) or _AADHAAR_RE_SPACED.search(txt)
    aad_num = None
    if aad_m2:
        aad_num = aad_m2.group(1).replace(' ', '')
    dob_m = _DOB_RE.search(txt)
    dob = dob_m.group(0) if dob_m else None
    gender = None
    if re.search(r'\bmale\b', lower):
        gender = 'Male'
    elif re.search(r'\bfemale\b', lower):
        gender = 'Female'
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    name_candidate = None
    candidates = []
    for ln in lines:
        lln = ln.lower()
        if any(t in lln for t in ['gov', 'government', 'aadhar', 'uidai', 'your', 'scan', 'aadhaar', 'india']):
            continue
        if re.search(r'\d', ln):
            continue
        clean = re.sub(r'[^A-Za-z\s\-\.]', '', ln).strip()
        if len(clean) >= 3:
            candidates.append(clean)
    for c in candidates:
        if ' ' in c and len(c.split()) <= 4:
            lowc = c.lower().replace(' ', '')
            if any(tok in lowc for tok in ['government', 'india', 'bharat', 'uidai']):
                continue
            name_candidate = c
            break
    if not name_candidate and candidates:
        name_candidate = max(candidates, key=lambda x: len(x))
    qr_found = len(micro_qr_list) > 0
    if is_like or aad_num or qr_found:
        is_like = True
    risk = 0
    if not qr_found:
        risk += 35
        notes.append('no_qr_found_on_aadhaar_like_doc' if is_like else 'no_qr_found')
    if not aad_num:
        risk += 40
        notes.append('uid_not_found' if is_like else 'no_uid_found')
    if not dob:
        risk += 40
        notes.append('dob_not_found')
    if name_candidate:
        if any(x in name_candidate.lower() for x in ['test', 'example', 'sample']):
            risk += 60
            notes.append('suspicious_name_token')
    if not aad_raw:
        risk += 45
        notes.append("uid_missing")
    elif is_suspicious_uid(aad_raw):
        risk += 45
        notes.append("uid_invalid_pattern")
    if len(txt) < 30:
        risk += 20
        notes.append("low_text_content")
    risk = max(0, min(100, risk))
    return {
        'is_aadhaar_like': bool(is_like),
        'aadhaar_fields': {'aadhaar_masked': aad_masked, 'dob': dob, 'gender': gender, 'name': name_candidate},
        'notes': notes,
        'qr_matches_text': None,
        'risk_score': int(risk)
    }

def detect_ids_combined(extracted_raw, micro_qr_list):
    pan = detect_pan_from_text(extracted_raw)
    aad = detect_aadhaar_from_text(extracted_raw, micro_qr_list)
    doc_type = 'unknown'
    report = {}
    if pan and pan.get('is_pan_like'):
        doc_type = 'pan'
        report = {'pan_report': pan}
        if aad and aad.get('is_aadhaar_like'):
            doc_type = 'both'
            report['aadhaar_report'] = aad
    elif aad and aad.get('is_aadhaar_like'):
        doc_type = 'aadhaar'
        report = {'aadhaar_report': aad}
    return {'doc_type': doc_type, 'doc_report': report}

def get_bytes_from_request_file(field_names=('capture', 'file')):
    for k in field_names:
        if k in request.files:
            f = request.files[k]
            data = f.read()
            return (f.filename or 'capture.jpg', data)
    return (None, None)

def mask_aadhaar(aad):
    if not aad or len(aad) != 12:
        return None
    return "XXXX XXXX " + aad[-4:]

def is_suspicious_uid(uid):
    if not uid or len(uid) != 12:
        return True
    if len(set(uid)) == 1:
        return True
    if uid in ["123456789012", "012345678901"]:
        return True
    if uid[:4] == uid[4:8] == uid[8:]:
        return True
    return False

# ---------- routes ----------
@app.route('/')
def index():
    if os.path.exists(os.path.join('ui', 'index.html')):
        return send_from_directory('ui', 'index.html')
    return "Credilock OCR+KYC API"

@app.route('/files', methods=['GET'])
def list_files():
    try:
        limit = int(request.args.get('limit', 50))
        name_q = request.args.get('name', '').strip()
        query = {}
        if name_q:
            query['filename'] = {'$regex': re.escape(name_q), '$options': 'i'}
        cursor = raw_col.find(query, {'filename': 1, 'upload_ts': 1, 'short_id': 1}).sort('upload_ts', -1).limit(limit)
        out = []
        for d in cursor:
            uid = str(d.get('_id'))
            up = d.get('upload_ts')
            up_s = up.isoformat() if isinstance(up, dt.datetime) else up
            out.append({'file_id': uid, 'short_id': d.get('short_id') or uid[:8], 'filename': d.get('filename'), 'upload_ts': up_s})
        return jsonify({'files': out}), 200
    except Exception:
        logger.exception("list_files error")
        return jsonify({'error': 'internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        filename, data = get_bytes_from_request_file(('file', 'capture'))
        if filename is None:
            return jsonify({'error': 'no file part'}), 400
        if filename == '':
            return jsonify({'error': 'no selected file'}), 400
        if not allowed_file(filename):
            return jsonify({'error': 'file type not supported'}), 400
        filename = secure_filename(filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(save_path):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(dt.datetime.utcnow().timestamp())}{ext}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(save_path, 'wb') as fh:
            fh.write(data)

        ext = filename.rsplit('.', 1)[1].lower()
        phash = None
        extracted_raw = ""
        extracted = ""
        ocr_engine = None
        micro_qr = []
        face_embedding = None
        face_backend = None
        face_enrolled = False

        if ext in ['png', 'jpg', 'jpeg']:
            phash = compute_phash_for_image(data)
            extracted_raw, words, ocr_engine = ocr_image(data)
            has_deva = is_devanagari_present(extracted_raw)
            extracted = filter_garbage_lines(extracted_raw, preserve_devanagari=has_deva)
            micro_qr = micro_qr_scan(data)
            emb_obj = enroll_best_embedding_from_bytes(data)
            if emb_obj and emb_obj.get('embedding'):
                face_backend = emb_obj.get('backend')
                face_embedding = normalize_vec(emb_obj.get('embedding'))
                face_enrolled = True
            else:
                face_enrolled = False
                face_embedding = None
                face_backend = None
        elif ext == 'pdf':
            extracted_raw = extract_text_from_pdf(save_path)
            has_deva = is_devanagari_present(extracted_raw)
            extracted = filter_garbage_lines(extracted_raw, preserve_devanagari=has_deva)
        elif ext in ['csv', 'json']:
            try:
                extracted_raw = data.decode('utf-8', errors='ignore')[:40000]
                has_deva = is_devanagari_present(extracted_raw)
                extracted = filter_garbage_lines(extracted_raw, preserve_devanagari=has_deva)
            except Exception:
                extracted = ""

        doc_detection = detect_ids_combined(extracted_raw, micro_qr)
        doc_type = doc_detection.get('doc_type')
        doc_report = doc_detection.get('doc_report')

        file_doc = {
            'filename': filename,
            'path': save_path,
            'upload_ts': dt.datetime.utcnow(),
            'extracted_preview': safe_preview(extracted),
            'extracted_raw_preview': safe_preview(extracted_raw),
            'phash': phash,
            'ocr_engine': ocr_engine,
            'ocr_words': [],
            'micro_qr': micro_qr,
            'face_enrolled': face_enrolled,
            'face_backend': face_backend,
            'face_embedding': serialize_for_mongo(face_embedding) if face_embedding is not None else None,
            'doc_type': doc_type,
            'doc_report': doc_report
        }

        res = raw_col.insert_one(serialize_for_mongo(file_doc))
        file_id = res.inserted_id
        short_id = str(file_id)[:8]
        raw_col.update_one({'_id': file_id}, {'$set': {'short_id': short_id}})

        ml_metrics = compute_and_persist_ml_metrics_for_file(file_id, file_doc)

        # ai_metrics — never fake confidence when pool is empty
        ai_metrics = {'avg_confidence': None, 'duplicates': 0, 'format_errors': 0, 'scanned': 0}
        try:
            note = ml_metrics.get('note', '')
            nearest_sim = ml_metrics.get('nearest_sim')
            if note in ('empty_pool', 'no_face_embedding', 'invalid_embedding', 'not_computed', 'nearest_failed', 'computed_no_candidate'):
                ai_metrics['avg_confidence'] = None
            elif nearest_sim is not None:
                ai_metrics['avg_confidence'] = (
                    ml_metrics.get('confidence') if ml_metrics.get('confidence') is not None
                    else face_confidence_from_similarity(float(nearest_sim))
                )
            else:
                ai_metrics['avg_confidence'] = None
            ai_metrics['duplicates'] = 1 if bool(ml_metrics.get('duplicate_exact')) else 0
        except Exception:
            ai_metrics['avg_confidence'] = None

        flags = []
        if not file_doc.get('extracted_preview', '').strip():
            flags.append({'type': 'Info', 'reason': 'No extractable text found', 'detail': ''})
        else:
            flags.append({'type': 'Info', 'reason': 'File processed', 'detail': ''})

        if doc_report:
            if doc_type == 'pan' and doc_report.get('pan_report'):
                pr = doc_report['pan_report']
                if pr.get('risk_score', 0) >= 40:
                    flags.append({'type': 'Fraud', 'reason': 'pan_high_risk', 'detail': f"risk={pr.get('risk_score')}"})
                if not pr['pan_fields'].get('name'):
                    flags.append({'type': 'Info', 'reason': 'pan_name_missing', 'detail': ''})
            if doc_type == 'aadhaar' and doc_report.get('aadhaar_report'):
                ar = doc_report['aadhaar_report']
                if ar.get('risk_score', 0) >= 50:
                    flags.append({'type': 'Fraud', 'reason': 'aadhaar_high_risk', 'detail': f"risk={ar.get('risk_score')}"})
            if doc_type == 'both':
                pr = doc_report.get('pan_report', {})
                ar = doc_report.get('aadhaar_report', {})
                if pr.get('risk_score', 0) >= 40 or ar.get('risk_score', 0) >= 50:
                    flags.append({'type': 'Fraud', 'reason': 'mixed_doc_high_risk', 'detail': f"pan={pr.get('risk_score')} aadhaar={ar.get('risk_score')}"})

        if flags:
            bulk = []
            for fl in flags:
                frec = dict(fl)
                frec['file_id'] = str(file_id)
                frec['created_at'] = dt.datetime.utcnow()
                bulk.append(serialize_for_mongo(frec))
            try:
                flags_col.insert_many(bulk)
            except Exception:
                pass

        response = {
            'message': 'uploaded',
            'file_id': str(file_id),
            'short_id': short_id,
            'extracted_text_preview': safe_preview(extracted, 4000),
            'extracted_raw_preview': safe_preview(extracted_raw, 4000),
            'has_devanagari': bool(is_devanagari_present(extracted_raw)),
            'ai_metrics': ai_metrics,
            'ml_metrics': ml_metrics,
            'face_enrolled': bool(face_enrolled),
            'doc_type': doc_type,
            'doc_report': doc_report
        }
        return jsonify(response), 200
    except Exception as e:
        logger.exception("upload exception: %s", e)
        return jsonify({'error': 'internal server error', 'detail': str(e)}), 500

@app.route('/ocr_debug', methods=['POST'])
def ocr_debug():
    try:
        filename, data = get_bytes_from_request_file(('file', 'capture'))
        if data is None:
            return jsonify({'error': 'file required'}), 400
        raw_text, words, engine = ocr_image(data)
        has_deva = is_devanagari_present(raw_text)
        cleaned = filter_garbage_lines(raw_text, preserve_devanagari=has_deva)
        return jsonify({'raw_text_preview': safe_preview(raw_text, 8000), 'cleaned_text_preview': safe_preview(cleaned, 8000), 'engine': engine, 'has_devanagari': has_deva}), 200
    except Exception as e:
        logger.exception("ocr_debug error: %s", e)
        return jsonify({'error': 'internal server error', 'detail': str(e)}), 500

@app.route('/ml_details', methods=['GET'])
def get_ml_details():
    file_id = request.args.get('file_id')
    if not file_id:
        return jsonify({'error': 'file_id required'}), 400
    doc = ml_col.find_one({'file_id': file_id}, sort=[('created_at', -1)])
    if not doc:
        return jsonify({'ml': None}), 200
    doc.pop('_id', None)
    if isinstance(doc.get('created_at'), dt.datetime):
        doc['created_at'] = doc['created_at'].isoformat()
    return jsonify({'ml': doc}), 200

@app.route('/flags', methods=['GET'])
def get_flags():
    file_id = request.args.get('file_id')
    if not file_id:
        return jsonify({'error': 'file_id required'}), 400
    try:
        page = int(request.args.get('page', '1'))
        page_size = int(request.args.get('page_size', '20'))
        query = {'file_id': file_id}
        total = flags_col.count_documents(query)
        skip = max(0, (page - 1) * page_size)
        cursor = flags_col.find(query).sort('created_at', ASCENDING).skip(skip).limit(page_size)
        out = []
        for d in cursor:
            d.pop('_id', None)
            if isinstance(d.get('created_at'), dt.datetime):
                d['created_at'] = d['created_at'].isoformat()
            out.append(d)
        return jsonify({'total': total, 'page': page, 'page_size': page_size, 'flags': out}), 200
    except Exception:
        logger.exception("flags exception")
        return jsonify({'error': 'internal server error'}), 500

@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json() or {}
        a = data.get('file_id_a')
        b = data.get('file_id_b')
        if not a or not b:
            return jsonify({'error': 'file_id_a and file_id_b required'}), 400
        try:
            fa = raw_col.find_one({'_id': ObjectId(a)})
        except Exception:
            fa = raw_col.find_one({'_id': a})
        try:
            fb = raw_col.find_one({'_id': ObjectId(b)})
        except Exception:
            fb = raw_col.find_one({'_id': b})
        if not fa or not fb:
            return jsonify({'error': 'one or both files not found'}), 404
        text_a = fa.get('extracted_preview', '') or ''
        text_b = fb.get('extracted_preview', '') or ''
        if not text_a or not text_b:
            return jsonify({'result': {'pass': False, 'score': 20, 'reasons': ['No extractable comparable text']}}), 200
        ta = set(re.findall(r'\w{4,}', text_a.lower()))
        tb = set(re.findall(r'\w{4,}', text_b.lower()))
        inter = ta.intersection(tb)
        jaccard = len(inter) / max(1, len(ta.union(tb)))
        score = 50 + jaccard * 60
        passed = score >= 65
        return jsonify({'result': {'pass': passed, 'score': round(score, 2), 'reasons': [f'text overlap Jaccard: {jaccard:.2f}']}}), 200
    except Exception:
        logger.exception("compare exception")
        return jsonify({'error': 'internal server error'}), 500

@app.route('/anchors', methods=['GET'])
def anchors():
    docs = list(ledger_col.find().sort('ts', -1).limit(50))
    out = []
    for d in docs:
        d['_id'] = str(d.get('_id'))
        if isinstance(d.get('ts'), dt.datetime):
            d['ts'] = d['ts'].isoformat()
        out.append(d)
    return jsonify(out)

@app.route('/vcard', methods=['POST'])
def vcard_route():
    try:
        data = request.get_json() or {}
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        file_id = data.get('file_id')
        if not name:
            return jsonify({'error': 'name required'}), 400
        txid = None
        signature = None
        pubkey_fp = None
        if file_id:
            try:
                doc = raw_col.find_one({'_id': ObjectId(file_id)})
            except Exception:
                doc = raw_col.find_one({'_id': file_id})
            if doc:
                p = doc.get('path')
                sha = None
                try:
                    with open(p, 'rb') as fh:
                        sha = hashlib.sha256(fh.read()).hexdigest()
                except Exception:
                    sha = None
                payload = {'file_id': str(doc.get('_id')), 'sha256': sha, 'meta': {'filename': doc.get('filename')}}
                tx = blockchain_anchor_signed(payload)
                txid = tx.get('txid')
                signature = tx.get('signature')
                pubkey_fp = tx.get('pubkey_fingerprint')
                try:
                    verified_col.insert_one({'file_id': str(doc.get('_id')), 'txid': txid, 'anchored_at': dt.datetime.utcnow(), 'payload': payload, 'signature': signature, 'pubkey_fingerprint': pubkey_fp})
                except Exception:
                    pass
        vcard_txt = generate_vcard(name, email, phone, txid)
        return jsonify({'vcard': vcard_txt, 'txid': txid, 'signature': signature, 'pubkey_fingerprint': pubkey_fp}), 200
    except Exception as e:
        logger.exception("vcard error: %s", e)
        return jsonify({'error': 'internal server error'}), 500

@app.route('/verify_anchor', methods=['GET'])
def verify_anchor():
    txid = request.args.get('txid')
    if not txid:
        return jsonify({'error': 'txid required'}), 400
    try:
        tx = ledger_col.find_one({'txid': txid})
        if not tx:
            return jsonify({'error': 'txid not found'}), 404
        signature = tx.get('signature')
        pub_b64 = tx.get('pubkey_pem')
        if not signature or not pub_b64:
            return jsonify({'verified': False, 'reason': 'no signature or pubkey stored'}), 200
        pub_pem_bytes = base64.b64decode(pub_b64)
        ok = verify_signature_for_txid(pub_pem_bytes, txid, signature)
        return jsonify({'verified': bool(ok), 'tx': tx}), 200
    except Exception:
        logger.exception("verify_anchor error")
        return jsonify({'error': 'internal server error'}), 500

# ---------- match_profile ----------
@app.route('/match_profile', methods=['POST'])
def match_profile():
    try:
        filename, data = get_bytes_from_request_file(('capture', 'file'))
        if data is None:
            return jsonify({'error': 'capture required'}), 400
        emb_obj = enroll_best_embedding_from_bytes(data)
        if not emb_obj:
            return jsonify({'error': 'no face found in capture or embedding failed'}), 400
        probe_emb = normalize_vec(emb_obj['embedding'])
        top_k = int(request.form.get('top_k', request.args.get('top_k', 3)))
        cursor = profiles_col.find({}).limit(2000)
        results = []
        for doc in cursor:
            pid = doc.get('profile_id')
            best_sim = -10.0
            if doc.get('embedding'):
                t = np.array(doc.get('embedding'), dtype=float)
                t = t / (np.linalg.norm(t) + 1e-9)
                sim = float(np.dot(probe_emb, t))
                if sim > best_sim:
                    best_sim = sim
            for shot in doc.get('embeddings', [])[:1000]:
                e = shot.get('embedding')
                if not e:
                    continue
                t = np.array(e, dtype=float)
                t = t / (np.linalg.norm(t) + 1e-9)
                sim = float(np.dot(probe_emb, t))
                if sim > best_sim:
                    best_sim = sim
            if best_sim < -0.9:
                continue
            conf = face_confidence_from_similarity(best_sim)
            results.append({'profile_id': pid, 'similarity': float(best_sim), 'confidence': conf, 'backend': doc.get('backend') or emb_obj.get('backend')})
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        return jsonify({'matches': results}), 200
    except Exception:
        logger.exception("match_profile error")
        return jsonify({'error': 'internal server error'}), 500

# ---------- FIXED match_biometric ----------
@app.route('/match_biometric', methods=['POST'])
def match_biometric():
    """
    Compare a probe image against ALL OTHER records in the DB.
    NEVER compares against itself (self-match was the 100% bug).

    Flow:
      1. Extract embedding from uploaded probe image.
      2. If file_id param given → compare probe against THAT specific file
         (used for doc-vs-selfie verification). Self-match blocked.
      3. If no file_id → scan entire DB, exclude self, return best match.
      4. If DB has no other records → return no_records_in_db state.
    """
    try:
        filename, data = get_bytes_from_request_file(('capture', 'file'))
        if data is None:
            return jsonify({'error': 'capture required (form key "capture" or "file")'}), 400

        # Step 1: get probe embedding
        emb_obj = enroll_best_embedding_from_bytes(data)
        if not emb_obj:
            return jsonify({
                'error': 'no_face_in_probe',
                'message': 'No face detected in the uploaded image. Please upload a clear photo with a visible face.',
                'has_face': False
            }), 400

        probe = normalize_vec(emb_obj['embedding'])
        probe_backend = emb_obj.get('backend')

        # Step 2: specific file_id comparison (doc vs selfie)
        file_id = request.form.get('file_id') or request.args.get('file_id')
        if file_id:
            try:
                target_oid = ObjectId(file_id)
            except Exception:
                target_oid = None

            # Block self-match: if probe came from the same file, reject
            # We detect this by checking if the uploaded file IS that file_id
            # (can't perfectly detect without hash, but we flag it conceptually)
            try:
                doc = raw_col.find_one({'_id': target_oid}) if target_oid else raw_col.find_one({'_id': file_id})
            except Exception:
                doc = None

            if not doc:
                return jsonify({'error': 'file not found'}), 404

            emb = doc.get('face_embedding')
            backend = doc.get('face_backend')

            if not emb:
                return jsonify({
                    'error': 'target_has_no_face',
                    'message': 'The target document has no enrolled face. Upload a document with a visible face photo.',
                    'has_face': False
                }), 400

            sim = match_embeddings(probe, emb, backend_hint=backend)
            conf = face_confidence_from_similarity(sim)

            # sim == 1.0 almost certainly means self-match
            if sim >= 0.9999:
                return jsonify({
                    'error': 'self_match_detected',
                    'message': 'The probe image appears to be identical to the target document. Upload a different (selfie) image to compare against the document.',
                    'self_match': True
                }), 400

            return jsonify({
                'match': True if (conf is not None and conf >= 60) else False,
                'similarity': round(sim, 4),
                'confidence': conf,
                'target_file': str(doc.get('_id')),
                'has_face': True
            }), 200

        # Step 3: scan all DB records, STRICTLY excluding self
        # Build exclusion: try to match the probe file by phash if available
        probe_phash = compute_phash_for_image(data)

        checked = 0
        best = None
        best_sim = -10.0
        self_matches_skipped = 0

        cursor = raw_col.find({'face_embedding': {'$exists': True, '$ne': None}}).limit(500)
        for doc in cursor:
            emb = doc.get('face_embedding')
            backend = doc.get('face_backend')
            if not emb:
                continue

            # Self-exclusion by phash: if this DB record's phash matches the probe, skip it
            if probe_phash and doc.get('phash'):
                dist = phash_distance(probe_phash, doc.get('phash'))
                if dist is not None and int(dist) <= 2:
                    # Almost certainly the same image
                    self_matches_skipped += 1
                    continue

            sim = match_embeddings(probe, emb, backend_hint=backend)

            # Additional self-match guard: skip near-perfect similarity
            if sim >= 0.9999:
                self_matches_skipped += 1
                continue

            checked += 1
            if sim > best_sim:
                best_sim = sim
                best = doc

        # Step 4: handle empty DB
        if checked == 0:
            return jsonify({
                'best_match': None,
                'checked': 0,
                'self_matches_skipped': self_matches_skipped,
                'no_records': True,
                'message': 'No other face records in the database. Upload more documents to enable face matching.'
            }), 200

        conf = face_confidence_from_similarity(best_sim)
        out = {
            'file_id': str(best.get('_id')),
            'filename': best.get('filename'),
            'similarity': round(float(best_sim), 4),
            'confidence': conf
        }
        return jsonify({
            'best_match': out,
            'checked': checked,
            'self_matches_skipped': self_matches_skipped,
            'has_face': True
        }), 200

    except Exception:
        logger.exception("match_biometric error")
        return jsonify({'error': 'internal server error'}), 500

@app.route('/match_capture', methods=['POST'])
def match_capture():
    try:
        filename, data = get_bytes_from_request_file(('capture', 'file'))
        if data is None:
            return jsonify({'error': 'no capture file provided'}), 400
        capture_ph = compute_phash_for_image(data)
        if not capture_ph:
            return jsonify({'best_match': None, 'error': 'capture phash failed'}), 200
        cursor = raw_col.find({'phash': {'$exists': True, '$ne': None}}).sort('upload_ts', -1).limit(1000)
        best_doc = None
        best_conf = -1
        best_dist = None
        checked = 0
        for doc in cursor:
            db_ph = doc.get('phash')
            if not db_ph:
                continue
            dist = phash_distance(capture_ph, db_ph)
            if dist is None:
                continue
            checked += 1
            try:
                d = int(dist)
            except Exception:
                d = 9999
            if d <= 4:
                conf = 98 - (d * 3)
            elif d <= 8:
                conf = 80 - ((d - 4) * 6)
            elif d <= 14:
                conf = 50 - ((d - 8) * 5)
            else:
                conf = max(3, 20 - (d - 14))
            conf = int(max(0, min(100, conf)))
            if conf > best_conf:
                best_conf = conf
                best_doc = doc
                best_dist = d
        if not best_doc:
            return jsonify({'best_match': None, 'checked': checked}), 200
        out = {'file_id': str(best_doc.get('_id')), 'filename': best_doc.get('filename'), 'phash': best_doc.get('phash'), 'distance': int(best_dist), 'confidence': int(best_conf)}
        return jsonify({'best_match': out, 'checked': checked}), 200
    except Exception:
        logger.exception("match_capture error")
        return jsonify({'error': 'internal server error'}), 500

@app.route('/enroll_profile', methods=['POST'])
def enroll_profile():
    try:
        profile_id = (request.form.get('profile_id') or request.args.get('profile_id') or '').strip()
        if not profile_id:
            return jsonify({'error': 'profile_id required'}), 400
        fname, data = get_bytes_from_request_file(('capture', 'file'))
        if data is None:
            return jsonify({'error': 'capture/file required'}), 400
        emb_obj = enroll_best_embedding_from_bytes(data)
        if not emb_obj or not emb_obj.get('embedding'):
            return jsonify({'error': 'no face found in capture or embedding failed'}), 400
        emb = normalize_vec(emb_obj['embedding'])
        backend = emb_obj.get('backend')
        shot_id = request.form.get('shot_id') or f"shot_{int(dt.datetime.utcnow().timestamp())}"
        label = request.form.get('label')
        profile = profiles_col.find_one({'profile_id': profile_id})
        now = dt.datetime.utcnow()
        shot_doc = {'shot_id': shot_id, 'created_at': now, 'embedding': serialize_for_mongo(emb), 'backend': backend, 'filename': fname}
        if profile is None:
            new_profile = {'profile_id': profile_id, 'label': label, 'created_at': now, 'updated_at': now, 'embeddings': [shot_doc], 'embedding': serialize_for_mongo(emb), 'backend': backend, 'meta': {}}
            try:
                profiles_col.insert_one(serialize_for_mongo(new_profile))
            except Exception:
                profiles_col.update_one({'profile_id': profile_id}, {'$push': {'embeddings': serialize_for_mongo(shot_doc)}, '$set': {'updated_at': now}})
            result = {'profile_id': profile_id, 'added_shot': shot_doc, 'profile_created': True}
        else:
            profiles_col.update_one({'profile_id': profile_id}, {'$push': {'embeddings': serialize_for_mongo(shot_doc)}, '$set': {'updated_at': now}})
            result = {'profile_id': profile_id, 'added_shot': shot_doc, 'profile_created': False}
        try:
            pdoc = profiles_col.find_one({'profile_id': profile_id})
            current_embed = pdoc.get('embedding') or (pdoc.get('embeddings') and pdoc.get('embeddings')[-1].get('embedding'))
            sim = None
            if current_embed:
                sim = float(np.dot(np.array(current_embed, dtype=float) / (np.linalg.norm(current_embed) + 1e-9), np.array(emb, dtype=float) / (np.linalg.norm(emb) + 1e-9)))
            confidence = face_confidence_from_similarity(sim) if sim is not None else None
        except Exception:
            sim = None
            confidence = None
        return jsonify({'result': result, 'similarity_to_profile': sim, 'confidence': confidence}), 200
    except Exception as e:
        logger.exception("enroll_profile error: %s", e)
        return jsonify({'error': 'internal server error', 'detail': str(e)}), 500

@app.route('/batch_enroll', methods=['POST'])
def batch_enroll():
    try:
        data = request.get_json(silent=True) or request.form or {}
        profile_id = (data.get('profile_id') or request.form.get('profile_id') or '').strip()
        if not profile_id:
            return jsonify({'error': 'profile_id required'}), 400
        method = (data.get('method') or request.form.get('method') or 'mean').lower()
        keep_shots = data.get('keep_shots', True)
        profile = profiles_col.find_one({'profile_id': profile_id})
        if not profile:
            return jsonify({'error': 'profile not found'}), 404
        shots = profile.get('embeddings', [])
        embs = []
        for s in shots:
            e = s.get('embedding')
            if e:
                try:
                    arr = np.array(e, dtype=float)
                    norm = arr / (np.linalg.norm(arr) + 1e-9)
                    embs.append(norm)
                except Exception:
                    pass
        if not embs:
            return jsonify({'error': 'no valid shot embeddings found for profile'}), 400
        X = np.stack(embs, axis=0)
        agg = np.median(X, axis=0) if method == 'median' else np.mean(X, axis=0)
        agg_norm = agg / (np.linalg.norm(agg) + 1e-9)
        now = dt.datetime.utcnow()
        update = {'embedding': serialize_for_mongo(agg_norm.tolist()), 'updated_at': now}
        if not keep_shots:
            update['embeddings'] = []
        profiles_col.update_one({'profile_id': profile_id}, {'$set': serialize_for_mongo(update)})
        return jsonify({'profile_id': profile_id, 'method': method, 'shots_count': len(embs), 'canonical_embedding_len': int(len(agg_norm)), 'updated_at': now.isoformat()}), 200
    except Exception as e:
        logger.exception("batch_enroll error: %s", e)
        return jsonify({'error': 'internal server error', 'detail': str(e)}), 500

@app.route('/final_verify', methods=['POST'])
def final_verify():
    try:
        data = request.get_json() or {}
        file_id = data.get('file_id')
        if not file_id:
            return jsonify({'error': 'file_id required'}), 400

        ml_doc = ml_col.find_one({'file_id': file_id}, sort=[('created_at', -1)])
        if not ml_doc:
            return jsonify({'error': 'no ml metrics available'}), 400

        ml = ml_doc.get('ml_metrics', {})
        nearest_sim = ml.get('nearest_sim')
        is_outlier = ml.get('is_outlier')
        duplicate_exact = ml.get('duplicate_exact', False)
        note = ml.get('note', '')

        try:
            doc = raw_col.find_one({'_id': ObjectId(file_id)})
        except Exception:
            doc = raw_col.find_one({'_id': file_id})
        if not doc:
            return jsonify({'error': 'file not found'}), 404
        doc_report = doc.get('doc_report') or {}

        # empty pool = insufficient data, not fraud
        if note == 'empty_pool':
            return jsonify({
                'passed': False,
                'fraud': False,
                'reasons': ['no_comparison_data_yet'],
                'message': 'No other documents in pool to compare against. Upload more documents before verification is meaningful.',
                'txid': None,
                'signature': None,
                'pubkey_fingerprint': None
            }), 200

        fraud = False
        reasons = []

        if is_outlier:
            fraud = True
            reasons.append('outlier_detected')

        if nearest_sim is None:
            fraud = True
            reasons.append('no_similarity_metric')
        else:
            if float(nearest_sim) < 0.50:
                fraud = True
                reasons.append('low_face_similarity')

        dtp = doc.get('doc_type')
        if dtp == 'aadhaar':
            ar = doc_report.get('aadhaar_report') if isinstance(doc_report, dict) else doc_report
            if ar and ar.get('risk_score', 0) >= 60:
                fraud = True
                reasons.append('aadhaar_high_risk')
            if ar and (not ar.get('aadhaar_fields', {}).get('aadhaar_number') or not ar.get('aadhaar_fields', {}).get('dob')):
                reasons.append('aadhaar_missing_fields')
        elif dtp == 'pan':
            pr = doc_report.get('pan_report') if isinstance(doc_report, dict) else doc_report
            if pr and pr.get('risk_score', 0) >= 50:
                fraud = True
                reasons.append('pan_high_risk')
            if pr and (not pr.get('pan_fields', {}).get('pan_number') or not pr.get('pan_fields', {}).get('name')):
                reasons.append('pan_missing_fields')
        elif dtp == 'both':
            pr = doc_report.get('pan_report', {})
            ar = doc_report.get('aadhaar_report', {})
            if pr.get('risk_score', 0) >= 50 or ar.get('risk_score', 0) >= 60:
                fraud = True
                reasons.append('mixed_doc_high_risk')

        if duplicate_exact:
            reasons.append('duplicate_exact_detected')

        if fraud:
            return jsonify({'passed': False, 'fraud': True, 'reasons': reasons, 'txid': None, 'signature': None, 'pubkey_fingerprint': None}), 200

        try:
            p = doc.get('path')
            with open(p, 'rb') as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            sha = None

        payload = {'file_id': file_id, 'sha256': sha, 'meta': {'filename': doc.get('filename')}}
        tx = blockchain_anchor_signed(payload)
        return jsonify({'passed': True, 'fraud': False, 'txid': tx.get('txid'), 'signature': tx.get('signature'), 'pubkey_fingerprint': tx.get('pubkey_fingerprint'), 'payload': payload}), 200

    except Exception as e:
        logger.exception("final_verify error: %s", e)
        return jsonify({'error': 'server error', 'detail': str(e)}), 500

if __name__ == '__main__':
    try:
        if PADDLE_AVAILABLE:
            ensure_paddle()
    except Exception:
        logger.exception("ensure_paddle failed")
    try:
        if EASY_AVAILABLE:
            ensure_easy()
    except Exception:
        logger.exception("ensure_easy failed")
    try:
        if FACENET_AVAILABLE:
            ensure_facenet()
    except Exception:
        logger.exception("ensure_facenet failed")
    logger.info("Starting Credilock. Paddle=%s Easy=%s Tesseract=%s FaceRec=%s FaceNet=%s Crypto=%s",
                PADDLE_AVAILABLE, EASY_AVAILABLE, TESSERACT_AVAILABLE, FACE_REC_AVAILABLE, FACENET_AVAILABLE, CRYPTO_AVAILABLE)
    app.run(host='0.0.0.0', port=5000, debug=False)