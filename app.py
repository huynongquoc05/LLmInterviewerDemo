from flask import Flask, request, jsonify, render_template, send_file
from pymongo import MongoClient
from LLMInterviewer3 import AdaptiveInterviewer
from gtts import gTTS
import os
import tempfile
import uuid
import threading
import time
from datetime import datetime, timedelta

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["interviewer_ai"]
collection = db["interview_results"]

# Khởi tạo Flask + Interviewer
app = Flask(__name__)
interviewer = AdaptiveInterviewer()

# Thư mục lưu trữ audio tạm thời
AUDIO_FOLDER = "temp_audio"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Dictionary để lưu trữ thông tin audio files
audio_cache = {}


def clean_old_audio_files():
    """Xóa các file audio cũ hơn 1 giờ"""
    try:
        now = datetime.now()
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if now - file_time > timedelta(hours=1):
                    os.remove(file_path)
                    print(f"Đã xóa file audio cũ: {filename}")
    except Exception as e:
        print(f"Lỗi khi xóa file audio cũ: {e}")

def cleanup_temp_files():
    """Xóa tất cả file tạm khi kết thúc chương trình"""
    try:
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Đã xóa file tạm: {filename}")
    except Exception as e:
        print(f"Lỗi khi xóa file tạm: {e}")


def create_audio_from_text(text, lang='vi'):
    """
    Tạo file audio từ text bằng gTTS

    Args:
        text (str): Văn bản cần chuyển thành giọng nói
        lang (str): Ngôn ngữ ('vi' cho tiếng Việt, 'en' cho tiếng Anh)

    Returns:
        str: Tên file audio được tạo
    """
    try:
        # Tạo tên file unique
        audio_id = str(uuid.uuid4())
        audio_filename = f"question_{audio_id}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

        # Xử lý text để tối ưu cho TTS
        # Loại bỏ các ký tự đặc biệt có thể gây lỗi
        clean_text = text.replace("🤖", "").strip()

        # Tạo audio bằng gTTS
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        tts.save(audio_path)

        # Lưu thông tin vào cache
        audio_cache[audio_id] = {
            'filename': audio_filename,
            'path': audio_path,
            'created_at': datetime.now(),
            'text': clean_text
        }

        print(f"✅ Đã tạo audio: {audio_filename}")
        return audio_id

    except Exception as e:
        print(f"❌ Lỗi tạo audio: {e}")
        return None


def detect_language(text):
    """
    Phát hiện ngôn ngữ của văn bản (đơn giản)
    """
    # Kiểm tra xem có ký tự tiếng Việt không
    vietnamese_chars = "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđ"
    vietnamese_chars += vietnamese_chars.upper()

    if any(char in text for char in vietnamese_chars):
        return 'vi'
    else:
        return 'en'


@app.route("/")
def index():
    # Cleanup old audio files khi khởi động
    clean_old_audio_files()
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    candidate = data["candidate"]
    topic = data["topic"]
    outline = data.get("outline", None)  # NEW: Lấy outline từ request

    # NEW: Truyền outline vào start_interview
    result = interviewer.start_interview(candidate, topic, outline)
    # Tạo audio cho câu hỏi đầu tiên
    if "question" in result:
        question_text = result["question"]
        lang = detect_language(question_text)

        # Tạo audio trong background thread để không block response
        def create_audio_async():
            audio_id = create_audio_from_text(question_text, lang)
            if audio_id:
                # Cập nhật result với audio_id (có thể lưu vào session hoặc cache)
                result["audio_id"] = audio_id

        # Tạo audio đồng bộ để trả về luôn
        audio_id = create_audio_from_text(question_text, lang)
        if audio_id:
            result["audio_id"] = audio_id
            result["audio_url"] = f"/audio/{audio_id}"

    return jsonify(result)


@app.route("/answer", methods=["POST"])
def answer():
    data = request.json
    candidate, answer = data["candidate"], data["answer"]

    result = interviewer.submit_answer(candidate, answer)

    # Nếu phỏng vấn chưa kết thúc và có câu hỏi tiếp theo
    if not result.get("finished") and "next_question" in result:
        question_text = result["next_question"]
        lang = detect_language(question_text)

        # Tạo audio cho câu hỏi tiếp theo
        audio_id = create_audio_from_text(question_text, lang)
        if audio_id:
            result["audio_id"] = audio_id
            result["audio_url"] = f"/audio/{audio_id}"

    # Lưu kết quả vào MongoDB nếu phỏng vấn kết thúc
    if result.get("finished"):
        insert_result = collection.insert_one(result["summary"])
        result["summary"]["_id"] = str(insert_result.inserted_id)

    return jsonify(result)


@app.route("/audio/<audio_id>")
def serve_audio(audio_id):
    """
    Endpoint để phục vụ file audio
    """
    try:
        if audio_id in audio_cache:
            audio_path = audio_cache[audio_id]['path']
            if os.path.exists(audio_path):
                return send_file(
                    audio_path,
                    mimetype="audio/mpeg",
                    as_attachment=False,
                    download_name=f"question_{audio_id}.mp3"
                )

        return jsonify({"error": "Audio file not found"}), 404

    except Exception as e:
        print(f"Lỗi khi phục vụ audio: {e}")
        return jsonify({"error": "Error serving audio file"}), 500


@app.route("/audio/info/<audio_id>")
def audio_info(audio_id):
    """
    Endpoint để lấy thông tin về audio file
    """
    if audio_id in audio_cache:
        info = audio_cache[audio_id].copy()
        info['created_at'] = info['created_at'].isoformat()
        info.pop('path', None)  # Không trả về đường dẫn file
        return jsonify(info)

    return jsonify({"error": "Audio not found"}), 404


@app.route("/test-tts", methods=["POST"])
def test_tts():
    """
    Endpoint để test tính năng TTS
    """
    data = request.json
    text = data.get("text", "Xin chào, đây là bài test text-to-speech")
    lang = data.get("lang", "vi")

    audio_id = create_audio_from_text(text, lang)
    if audio_id:
        return jsonify({
            "success": True,
            "audio_id": audio_id,
            "audio_url": f"/audio/{audio_id}",
            "text": text,
            "language": lang
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to create audio"
        }), 500


# Background task để dọn dẹp file audio định kỳ
def cleanup_scheduler():
    """
    Chạy cleanup mỗi 30 phút
    """
    while True:
        time.sleep(30 * 60)  # 30 phút
        clean_old_audio_files()

        # Dọn dẹp cache
        now = datetime.now()
        expired_keys = []
        for audio_id, info in audio_cache.items():
            if now - info['created_at'] > timedelta(hours=1):
                expired_keys.append(audio_id)

        for key in expired_keys:
            audio_cache.pop(key, None)

        if expired_keys:
            print(f"Đã xóa {len(expired_keys)} audio cache entries")


if __name__ == "__main__":
    # Khởi động background cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    cleanup_thread.start()

    print("🚀 Server đang khởi động...")
    print("🔊 Text-to-Speech đã được kích hoạt")
    print("📁 Audio files được lưu tại:", AUDIO_FOLDER)

    app.run(debug=True, threaded=True)
    #Xóa file tạm khi kết thúc
    cleanup_temp_files()

