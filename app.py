import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import requests
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, request, jsonify

from transformers import AutoTokenizer
from transformers.models.bert.modeling_tf_bert import TFBertModel

# CONFIG
MODEL_DIR = "models"
MODEL_PATH = "models/best_model.h5"

SCALER_PATH = "models/scaler.pkl"
NUMERIC_COLUMNS_PATH = "models/numeric_columns.pkl"
TOKENIZER_PATH = "models/tokenizer"
TOKENIZER_PATH = "models/indobert_tokenizer"

MODEL_URL = os.environ.get("MODEL_URL")

MAX_LEN = 128

label_mapping = {
    0: "Banjir",
    1: "Jalan Rusak",
    2: "Sampah"
}


# DOWNLOAD MODEL
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("Model sudah ada.")
        return

    if not MODEL_URL:
        raise Exception("MODEL_URL belum diisi di environment variable.")

    print("Downloading model from Hugging Face...")

    response = requests.get(MODEL_URL, stream=True)

    if response.status_code != 200:
        raise Exception(f"Gagal download model. Status code: {response.status_code}")

    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    print("Model berhasil didownload.")


@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features):
        score = tf.nn.tanh(self.W(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# LOAD ASSETS
download_model()

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionLayer": AttentionLayer,
        "TFBertModel": TFBertModel
    },
    compile=False
)

scaler = joblib.load(SCALER_PATH)
NUMERIC_COLUMNS = joblib.load(NUMERIC_COLUMNS_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print("Semua asset berhasil dimuat.")

# PREDICTION
def predict_citizencare(text):
    text = str(text).strip()

    if text == "":
        return {
            "error": "Deskripsi laporan tidak boleh kosong."
        }

    lower_text = text.lower()

    urgency_keywords = [
        "darurat", "parah", "bahaya", "berbahaya",
        "membahayakan", "tinggi", "rusak berat",
        "besar", "fatal", "mendesak", "segera",
        "terendam", "ambles", "hujan deras"
    ]

    urgency_score = 0

    for keyword in urgency_keywords:
        if keyword in lower_text:
            urgency_score += 15

    urgency_score = min(urgency_score, 100)

    numeric_data = pd.DataFrame([{
        "char_length": len(text),
        "word_count": len(text.split()),
        "has_urgency_keyword": 1 if urgency_score > 0 else 0,
        "urgency_score": urgency_score,

        "has_banjir_keyword": 1 if any(
            keyword in lower_text
            for keyword in ["banjir", "terendam", "genangan", "air tinggi", "hujan deras"]
        ) else 0,

        "has_jalan_keyword": 1 if any(
            keyword in lower_text
            for keyword in ["jalan", "berlubang", "aspal", "ambles", "rusak", "retak"]
        ) else 0,

        "has_sampah_keyword": 1 if any(
            keyword in lower_text
            for keyword in ["sampah", "limbah", "tumpukan", "menumpuk", "bau"]
        ) else 0
    }])

    numeric_data = numeric_data.reindex(
        columns=NUMERIC_COLUMNS,
        fill_value=0
    )

    numeric_scaled = scaler.transform(numeric_data).astype("float32")

    tokens = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

    predictions = model.predict(
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "numeric_input": numeric_scaled
        },
        verbose=0
    )

    pred_class_probs = predictions[0]
    pred_severity = predictions[1]

    predicted_class = int(np.argmax(pred_class_probs[0]))

    confidence = round(
        float(np.max(pred_class_probs[0]) * 100),
        2
    )

    model_label = label_mapping.get(predicted_class, "Unknown")

    rule_scores = {
        "Banjir": sum(
            keyword in lower_text
            for keyword in ["banjir", "terendam", "genangan", "air tinggi", "hujan deras"]
        ),
        "Jalan Rusak": sum(
            keyword in lower_text
            for keyword in ["jalan", "berlubang", "aspal", "ambles", "rusak", "retak"]
        ),
        "Sampah": sum(
            keyword in lower_text
            for keyword in ["sampah", "limbah", "tumpukan", "menumpuk", "bau"]
        )
    }

    best_rule_label = max(rule_scores, key=rule_scores.get)

    if rule_scores[best_rule_label] > 0:
        final_label = best_rule_label
    else:
        final_label = model_label

    model_severity_score = round(
        float(pred_severity[0][0] * 100),
        2
    )

    severe_keywords = [
        "besar", "parah", "berbahaya", "membahayakan",
        "tinggi", "darurat", "segera", "fatal",
        "rusak parah", "air tinggi", "hujan deras"
    ]

    keyword_bonus = 0

    for keyword in severe_keywords:
        if keyword in lower_text:
            keyword_bonus += 12

    severity_score = max(
        model_severity_score,
        urgency_score,
        keyword_bonus
    )

    severity_score = round(min(severity_score, 100), 2)

    if severity_score >= 80:
        severity_label = "Sangat Parah"
        recommended_action = "Segera kirim tim darurat dan lakukan penanganan prioritas tinggi."
    elif severity_score >= 60:
        severity_label = "Parah"
        recommended_action = "Perlu penanganan cepat dari petugas terkait."
    elif severity_score >= 40:
        severity_label = "Sedang"
        recommended_action = "Masukkan ke daftar pemantauan dan jadwalkan perbaikan."
    elif severity_score >= 20:
        severity_label = "Ringan"
        recommended_action = "Lakukan pengecekan lapangan jika diperlukan."
    else:
        severity_label = "Sangat Ringan"
        recommended_action = "Kondisi masih relatif aman dan dapat dipantau."

    probabilities = {}

    for index, prob in enumerate(pred_class_probs[0]):
        label_name = label_mapping.get(index, f"class_{index}")
        probabilities[label_name] = round(float(prob * 100), 2)

    return {
        "jenis_bencana": final_label,
        "deskripsi_laporan": text,
        "tingkat_kerusakan": severity_label,
        "skor_kerusakan": severity_score,
        "confidence": confidence,
        "urgency_keyword_score": urgency_score,
        "model_prediction": model_label,
        "hybrid_prediction": final_label,
        "model_severity_score": model_severity_score,
        "probabilities": probabilities,
        "recommended_action": recommended_action
    }

# FLASK APP
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "CitizenCare API Running",
        "status": "success"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body harus JSON."
            }), 400

        text = data.get("text") or data.get("deskripsi_laporan")

        if not text:
            return jsonify({
                "error": "Field 'text' atau 'deskripsi_laporan' wajib diisi."
            }), 400

        result = predict_citizencare(text)

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )