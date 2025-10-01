# AdaptiveInterviewer: AI Interviewer với State Machine thông minh
import datetime

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from GetApikey import loadapi

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


# =======================
# 1. Enums & Data Classes
# =======================

class Level(Enum):
    YEU = "yeu"  # <5
    TRUNG_BINH = "trung_binh"  # 5-6.5
    KHA = "kha"  # 6.5-8
    GIOI = "gioi"  # 8-9
    XUAT_SAC = "xuat_sac"  # 9-10


class QuestionDifficulty(Enum):
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


@dataclass
class QuestionAttempt:
    question: str
    answer: str
    score: float
    analysis: str
    difficulty: QuestionDifficulty
    timestamp: str


@dataclass
class InterviewState:
    candidate_name: str
    profile: str
    level: Level
    topic: str
    current_difficulty: QuestionDifficulty
    attempts_at_current_level: int
    max_attempts_per_level: int
    total_questions_asked: int
    max_total_questions: int
    upper_level_reached: int
    history: List[QuestionAttempt]
    is_finished: bool
    final_score: Optional[float] = None


# =======================
# 2. Configuration & Thresholds
# =======================

class InterviewConfig:
    # Thresholds để quyết định next step
    THRESHOLD_HIGH = 7.0  # >= 7: chuyển lên khó hơn
    THRESHOLD_LOW = 4.0  # < 4: giảm xuống dễ hơn

    # Limits
    MAX_ATTEMPTS_PER_LEVEL = 2
    MAX_TOTAL_QUESTIONS = 8
    MAX_UPPER_LEVEL = 2  # max level có thể đạt được

    # Difficulty progression mapping
    DIFFICULTY_MAP = {
        Level.YEU: [QuestionDifficulty.VERY_EASY, QuestionDifficulty.EASY],
        Level.TRUNG_BINH: [QuestionDifficulty.EASY, QuestionDifficulty.EASY],
        Level.KHA: [QuestionDifficulty.MEDIUM, QuestionDifficulty.HARD],
        Level.GIOI: [QuestionDifficulty.MEDIUM, QuestionDifficulty.VERY_HARD],
        Level.XUAT_SAC: [QuestionDifficulty.HARD, QuestionDifficulty.VERY_HARD],
    }


# =======================
# 3. Utility Functions
# =======================

def classify_level_from_score(score_40: float) -> Level:
    """Phân loại level dựa trên điểm 40%"""
    if score_40 < 5.0:
        return Level.YEU
    elif score_40 <= 6.5:
        return Level.TRUNG_BINH
    elif score_40 <= 8.0:
        return Level.KHA
    elif score_40 <= 9.0:
        return Level.GIOI
    else:
        return Level.XUAT_SAC


def get_initial_difficulty(level: Level) -> QuestionDifficulty:
    """Lấy độ khó ban đầu cho level"""
    return InterviewConfig.DIFFICULTY_MAP[level][0]


def get_next_difficulty(current: QuestionDifficulty, action: str) -> QuestionDifficulty:
    """Tính độ khó tiếp theo dựa trên action (harder/same/easier)"""
    difficulties = list(QuestionDifficulty)
    current_idx = difficulties.index(current)

    if action == "harder" and current_idx < len(difficulties) - 1:
        return difficulties[current_idx + 1]
    elif action == "easier" and current_idx > 0:
        return difficulties[current_idx - 1]
    else:  # same or can't change
        return current


import json, re

import re
import json

def _clean_and_parse_json_response(raw_text: str, expected_keys: list[str] = None) -> dict:
    """
    Parse JSON từ LLM, xử lý cả khi trong string có code block hoặc text thừa.
    Nếu có code block, sẽ nối code vào trường 'question' thay vì tách riêng.
    """
    if not raw_text:
        return {}

    text = raw_text.strip()

    # 1) Tìm code block (java, python...)
    code_match = re.search(r"```(?:[a-zA-Z0-9]+)?\s*(.*?)\s*```", text, flags=re.S)
    code_snippet = code_match.group(1).strip() if code_match else None

    # 2) Thử parse JSON object trong toàn bộ text
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        # Loại bỏ code fences
        candidate = re.sub(r"```[a-zA-Z]*", "", candidate)
        candidate = candidate.replace("```", "")

        # Escape xuống dòng trong string
        def _escape_newlines_in_strings(match):
            inner = match.group(0)
            return inner.replace("\n", "\\n")
        candidate = re.sub(r'\".*?\"', _escape_newlines_in_strings, candidate, flags=re.S)

        try:
            parsed = json.loads(candidate)
            if expected_keys:
                parsed = {k: v for k, v in parsed.items() if k in expected_keys}
            # Nối code vào question nếu có
            if code_snippet and "question" in parsed:
                parsed["question"] += "\n\n" + code_snippet
            return parsed
        except Exception as e:
            print(f"⚠️ JSON parse error after cleaning: {e}")

    # 3) Nếu thất bại, fallback sang parse single question
    return {"question": _clean_and_parse_single_question(text, code_snippet)}


def _clean_and_parse_single_question(raw_text: str, code_snippet: str = None) -> str:
    """
    Input: raw_text từ LLM (có thể kèm ```json``` hoặc lộn xộn)
    Output: 1 string câu hỏi sạch, kèm code snippet nếu có
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1) Nếu có code fence JSON
    code_fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if code_fence_match:
        text = code_fence_match.group(1).strip()

    # 2) Thử parse JSON object
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "question" in parsed:
                q = _sanitize_question(parsed["question"])
                if code_snippet:
                    q += "\n\n" + code_snippet
                return q
        except Exception:
            pass

    # 3) Nếu thất bại, thử tìm chuỗi trong ngoặc kép
    quoted = re.findall(r'"([^"]{10,})"', text, flags=re.S)
    if quoted:
        q = _sanitize_question(quoted[0])
        if code_snippet:
            q += "\n\n" + code_snippet
        return q

    # 4) Fallback: lấy dòng dài nhất làm câu hỏi
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    if lines:
        q = _sanitize_question(max(lines, key=len))
        if code_snippet:
            q += "\n\n" + code_snippet
        return q

    return code_snippet or ""


def _sanitize_question(q: str) -> str:
    """Làm sạch câu hỏi: bỏ backticks, quotes, số thứ tự..."""
    s = str(q).strip()
    s = re.sub(r'^[`\"]+|[`\"]+$', '', s).strip()
    s = re.sub(r'^\s*"\s*', '', s)
    s = re.sub(r'^\s*\(?\d+\)?[\).\s:-]+\s*', '', s)
    s = s.rstrip(",;}]")
    return s.strip()


# =======================
# 4. Core Interviewer Class
# =======================

class AdaptiveInterviewer:
    def __init__(self):
        # Load components
        self.api_key = loadapi()
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
        self.cv_db = FAISS.load_local("vector_db_csv", self.embeddings, allow_dangerous_deserialization=True)
        self.knowledge_db = FAISS.load_local("vector_db2chunk_nltk", self.embeddings,
                                             allow_dangerous_deserialization=True)
        self.retriever = self.knowledge_db.as_retriever(search_kwargs={"k": 5})
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.5
        )
        # === New: conversation memory (simple list) ===
        self.memory: list[dict] = []
        self.max_memory_turns = 6   # chỉ giữ 6 lượt gần nhất
        self.sessions: dict[str, InterviewState] = {}  # giữ state theo candidate_name
        self.knowledge_text=None

    # ============ Memory Helpers ============
    def add_to_memory(self, role: str, content: str):
        """Thêm một đoạn hội thoại vào memory."""
        self.memory.append({"role": role, "content": content})
        self.memory = self.memory[-self.max_memory_turns:]  # cắt bớt nếu quá dài

    def build_history_prompt(self) -> str:
        """Ghép memory thành đoạn hội thoại để truyền vào LLM."""
        if not self.memory:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.memory])

    def load_candidate_profile(self, candidate_name: str) -> tuple[str, Level]:
        """Load hồ sơ và phân loại level"""
        profile_docs = self.cv_db.similarity_search(candidate_name, k=1)
        if not profile_docs:
            raise ValueError(f"Không tìm thấy hồ sơ cho {candidate_name}")

        profile_content = profile_docs[0].page_content

        # Extract điểm 40% từ profile (giả sử có format chuẩn)
        score_match = re.search(r'Điểm 40%[:\s]+([0-9.]+)', profile_content)
        if score_match:
            score_40 = float(score_match.group(1))
            level = classify_level_from_score(score_40)
        else:
            # Fallback: dùng LLM để classify
            level = self._classify_level_with_llm(profile_content)

        return profile_content, level

    def build_knowledge_context(self, topic: str, outline: list[str] | None = None) -> str:
        """
        Tạo ngữ cảnh kiến thức dựa trên topic và optional outline.
        - Nếu có outline: search nhiều query (topic + từng mục outline).
        - Nếu không có: search theo topic.
        Trả về: text ghép nối từ các tài liệu.
        """
        results = []

        if outline and len(outline) > 0:
            # Multiple query
            for item in outline:
                query = f"{topic} {item}"
                docs = self.retriever.invoke(query)
                results.extend(docs)
        else:
            # Single query
            docs = self.retriever.invoke(topic)
            results.extend(docs)

        # Loại trùng lặp (theo page_content)
        seen = set()
        unique_docs = []
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        knowledge_text = "\n\n".join([doc.page_content for doc in unique_docs])
        self.knowledge_text=knowledge_text

    def _classify_level_with_llm(self, profile: str) -> Level:
        """Fallback method để classify level bằng LLM"""
        classify_prompt = f"""
        Bạn là một Interviewer AI và đang chuẩn bị phỏng vấn bài thi vấn đáp của 1 thí sinh .
        Phân loại trình độ thí sinh theo điểm 40%: Yếu (<5), Trung bình (5-6.5), Khá (6.5-8), Giỏi (8-9), Xuất sắc (9-10).

        Hồ sơ: {profile}

        Trả về JSON: {{"level": "yeu|trung_binh|kha|gioi|xuat_sac"}}
        """
        result = self.llm.invoke(classify_prompt)
        parsed = _clean_and_parse_json_response(result, ["level"])
        level_str = parsed.get("level", "trung_binh")

        # Convert to enum
        level_mapping = {
            "yeu": Level.YEU,
            "trung_binh": Level.TRUNG_BINH,
            "kha": Level.KHA,
            "gioi": Level.GIOI,
            "xuat_sac": Level.XUAT_SAC
        }
        return level_mapping.get(level_str, Level.TRUNG_BINH)

    def generate_question(self, topic: str, difficulty: QuestionDifficulty, context: str = "") -> str:
        """Generate câu hỏi theo topic và độ khó"""
        # knowledge_context = self.retriever.invoke(f"{topic} {difficulty.value}")
        # knowledge_text = "\n\n".join([doc.page_content for doc in knowledge_context])
        history_text = self.build_history_prompt()  # Lấy lịch sử hội thoại
        #print("history_text:", history_text)
        difficulty_descriptions = {
            QuestionDifficulty.VERY_EASY: "rất cơ bản, định nghĩa đơn giản",
            QuestionDifficulty.EASY: "cơ bản, ví dụ thực tế",
            QuestionDifficulty.MEDIUM: "trung cấp, ứng dụng thực tế",
            QuestionDifficulty.HARD: "nâng cao, phân tích sâu",
            QuestionDifficulty.VERY_HARD: "rất khó, tổng hợp kiến thức"
        }

        generate_prompt = f"""
        Bạn là một Interviewer AI.
        Đây là lịch sử hội thoại gần đây giữa bạn và thí sinh:
        {history_text}

        Tạo 1 câu hỏi phỏng vấn Java về chủ đề "{topic}" với độ khó "{difficulty_descriptions[difficulty]}".

        QUAN TRỌNG:
        - Chỉ sử dụng kiến thức trong phần TÀI LIỆU THAM KHẢO dưới đây để tạo câu hỏi.
        -các ví dụ code thì bạn có thể sáng tạo nhưng phải dựa vào kiến thức trong tài liệu tham khảo
        - Nếu tài liệu tham khảo trống hoặc không chứa thông tin liên quan đến "{topic}", 
          thì KHÔNG được tự sáng tạo câu hỏi, hãy trả về JSON:
          {{"question": "Không đủ dữ liệu để tạo câu hỏi."}}

        TÀI LIỆU THAM KHẢO:
        {self.knowledge_text if self.knowledge_text else "Không có tài liệu"}

        YÊU CẦU:
        - Câu hỏi phải rõ ràng, cụ thể, phù hợp với độ khó.
        - TUYỆT ĐỐI KHÔNG được dùng kiến thức ngoài tài liệu tham khảo.
        - Ví dụ code được phép dùng nhưng phải dựa vào nội dung trong tài liệu.
        - Có cân nhắc lịch sử hội thoại giữa bạn và thí sinh để câu hỏi mạch lạc hơn.
        - Văn phong tự nhiên, hạn chế lặp lại cụm từ như "theo tài liệu tham khảo",“tài liệu đề cập”..., mà thay thế bằng “những gì tôi được  biết”….
        -Để văn phong tự nhiên hơn , trước khi đưa ra nội dung câu hỏi mới, bạn hãy dành 1 lời khen, nếu thí sinh trả lời tốt câu trước đó rồi hãy đưa ra nội dung câu hỏi, còn không thì thôi.

        Đầu ra: 
        - Trả về DUY NHẤT một object JSON có dạng: {{"question": " lời khen (nếu có) + câu hỏi..."}}
        - KHÔNG kèm lời chào, giải thích hay code fence (```).
        """
        print("tạo ra câu hỏi với độ khó:", difficulty.value)
        result = self.llm.invoke(generate_prompt)
        print("Raw LLM output for question generation:", result)
        # print(result)
        parsed = _clean_and_parse_json_response(result, ["question"])
        # print("self.knowledge_text", self.knowledge_text)

        print("Generated question:", parsed)
        # print(parsed)
        self.add_to_memory("interviewer", parsed.get("question", "Hãy giải thích về Java?"))
         # Thêm câu hỏi vào memory
        # print(self.memory)
        return parsed.get("question", "Hãy giải thích về Java?")

    def evaluate_answer(self, question: str, answer: str, topic: str) -> tuple[float, str]:
        """Đánh giá câu trả lời và trả về (score, analysis)"""
        # Lấy ngữ cảnh kiến thức
        knowledge_text = self.knowledge_text if self.knowledge_text else "Không có tài liệu"
        history_text = self.build_history_prompt()
        eval_prompt = f"""
        Đây là lịch sử hội thoại gần đây:
        {history_text}

        Nhiệm vụ: Chấm điểm câu trả lời phỏng vấn Java (0-10 điểm).

        Câu hỏi: {question}
        Câu trả lời: {answer}

        TÀI LIỆU THAM KHẢO (nguồn duy nhất để chấm điểm):
        {knowledge_text}

        QUY TẮC BẮT BUỘC:
        - Chỉ được sử dụng kiến thức có trong TÀI LIỆU THAM KHẢO để đánh giá.
        - KHÔNG được thêm, suy diễn hay viện dẫn kiến thức ngoài tài liệu (ví dụ: "reference types" nếu tài liệu không đề cập).
        - Nếu câu trả lời có phần vượt ra ngoài tài liệu, thì KHÔNG được coi đó là sai. Chỉ cần chấm dựa trên những gì tài liệu có.
        - Nếu tài liệu không đủ thông tin để khẳng định đúng/sai, hãy chấm ở mức trung lập (5/10) và ghi nhận xét: "Tài liệu không đề cập, không thể đánh giá đầy đủ."

        ĐẦU RA BẮT BUỘC:
        Trả về JSON duy nhất theo dạng:
        {{
          "score": <số từ 0-10>,
          "analysis": "<nhận xét ngắn gọn, chỉ dựa trên tài liệu>"
        }}
        """


        # print(history_text)
        result = self.llm.invoke(eval_prompt)
        parsed = _clean_and_parse_json_response(result, ["score", "analysis"])

        score = float(parsed.get("score", 5.0))
        analysis = parsed.get("analysis", "Không có nhận xét")
        # === Cập nhật memory ===
        self.add_to_memory("student", answer)
        self.add_to_memory("interviewer", f"📊 Điểm: {score}/10 - {analysis}")
        # print("current memory:", self.memory)
        return score, analysis

    def decide_next_action(self, score: float, state: InterviewState) -> str:
        """Policy Engine: quyết định action tiếp theo"""
        if score >= InterviewConfig.THRESHOLD_HIGH:
            return "harder"
        elif score >= InterviewConfig.THRESHOLD_LOW:
            return "same"
        else:
            return "easier"

    def update_state_after_question(self, state: InterviewState,
                                    question: str, answer: str,
                                    score: float, analysis: str) -> None:
        """Update state sau mỗi câu hỏi (không tạo thêm attempt mới để tránh nhân đôi)"""
        # Cập nhật bộ đếm
        state.total_questions_asked += 1

        # Quyết định action
        action = self.decide_next_action(score, state)

        if action == "harder":

            state.upper_level_reached += 1
            print('số level chẩn bị lên:', state.upper_level_reached)
            if state.upper_level_reached <= InterviewConfig.MAX_UPPER_LEVEL:
                state.current_difficulty = get_next_difficulty(state.current_difficulty, "harder")
                state.attempts_at_current_level = 0
            else:
                # Đã vượt giới hạn nâng cấp
                state.is_finished = True

        elif action == "same":
            # Giữ nguyên độ khó, tăng số lần ở level này
            state.attempts_at_current_level += 1

        else:  # easier
            # Hạ độ khó, tăng số lần
            state.current_difficulty = get_next_difficulty(state.current_difficulty, "easier")
            state.attempts_at_current_level += 1
            state.upper_level_reached = max(0, state.upper_level_reached - 1)

        # Kiểm tra điều kiện kết thúc
        if (state.attempts_at_current_level >= InterviewConfig.MAX_ATTEMPTS_PER_LEVEL or
                state.total_questions_asked >= InterviewConfig.MAX_TOTAL_QUESTIONS or
                state.is_finished):
            state.is_finished = True
            scores = [attempt.score for attempt in state.history if attempt.score > 0]
            state.final_score = sum(scores) / len(scores) if scores else 0.0

    def start_interview(self, candidate_name: str, topic: str,outline: list[str] | None = None) -> Dict:
        # 1. Load profile + phân loại level
        profile, level = self.load_candidate_profile(candidate_name)
        initial_difficulty = get_initial_difficulty(level)
        self.build_knowledge_context(topic, outline)
        print ('knowledge_text:', self.knowledge_text)
        # 2. Khởi tạo state
        state = InterviewState(
            candidate_name=candidate_name,
            profile=profile,
            level=level,
            topic=topic,
            current_difficulty=initial_difficulty,
            attempts_at_current_level=0,
            max_attempts_per_level=InterviewConfig.MAX_ATTEMPTS_PER_LEVEL,
            total_questions_asked=0,
            max_total_questions=InterviewConfig.MAX_TOTAL_QUESTIONS,
            history=[],
            is_finished=False
            , upper_level_reached=0,

        )

        # 3. Sinh câu hỏi đầu tiên
        question = self.generate_question(topic, state.current_difficulty, "Bắt đầu phỏng vấn")

        # 4. Lưu vào history (chưa có answer, score, analysis)
        state.history.append(QuestionAttempt(
            question=question,
            answer="",
            score=0.0,
            analysis="(pending answer)",
            difficulty=state.current_difficulty,
            timestamp=datetime.datetime.now().isoformat()
        ))

        # 5. Lưu state vào sessions
        self.sessions[candidate_name] = state

        return {
            "candidate": candidate_name,
            "topic": topic,
            "profile": profile,
            "level": level.value,
            "question": question,
            "difficulty": state.current_difficulty.value,
        }

    def submit_answer(self, candidate_name: str, answer: str) -> Dict:
        # 1. Lấy state từ sessions
        state = self.sessions.get(candidate_name)
        if not state:
            return {"error": "Interview not started"}

        # 2. Lấy câu hỏi cuối cùng trong history
        if not state.history:
            return {"error": "No question found in history"}
        last_attempt = state.history[-1]
        last_question = last_attempt.question

        # 3. Chấm điểm
        score, analysis = self.evaluate_answer(last_question, answer, state.topic)
        print('answer:', answer)
        print(f"Evaluated answer. Score: {score}, Analysis: {analysis}")
        # 4. Cập nhật lại attempt cuối cùng
        last_attempt.answer = answer
        last_attempt.score = score
        last_attempt.analysis = analysis

        # 5. Update state (điểm, số lần, độ khó…)
        self.update_state_after_question(state, last_question, answer, score, analysis)

        # 6. Nếu kết thúc
        if state.is_finished:
            summary = self.generate_summary(state)
            return {"finished": True, "summary": summary}

        # 7. Nếu chưa kết thúc → sinh câu hỏi mới & append vào history
        next_question = self.generate_question(
            state.topic,
            state.current_difficulty,
            f"Đã hỏi {state.total_questions_asked} câu"
        )

        state.history.append(QuestionAttempt(
            question=next_question,
            answer="",
            score=0.0,
            analysis="(pending answer)",
            difficulty=state.current_difficulty,
            timestamp=datetime.datetime.now().isoformat()
        ))

        return {
            "finished": False,
            "score": score,
            "analysis": analysis,
            "next_question": next_question,
            "difficulty": state.current_difficulty.value,
        }

    def generate_summary(self, state: InterviewState) -> Dict:
        """Generate final interview summary"""
        print("\n" + "=" * 50)
        print("📝 TỔNG KẾT PHỎNG VẤN")
        print("=" * 50)

        summary = {
            "candidate_info": {
                "name": state.candidate_name,
                "profile": state.profile,
                "classified_level": state.level.value
            },
            "interview_stats": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_questions": len(state.history),
                "final_score": state.final_score,
                "topic": state.topic
            },
            "question_history": []
        }

        for i, attempt in enumerate(state.history, 1):
            q_info = {

                "question_number": i,
                "difficulty": attempt.difficulty.value,
                "question": attempt.question,
                "answer": attempt.answer,
                "score": attempt.score,
                "analysis": attempt.analysis
            }
            summary["question_history"].append(q_info)

            print(f"\nCâu {i} ({attempt.difficulty.value}):")
            print(f"Q: {attempt.question}")
            print(f"A: {attempt.answer}")
            print(f"Score: {attempt.score}/10 - {attempt.analysis}")

        print(f"\n🏆 ĐIỂM TỔNG KẾT: {state.final_score:.1f}/10")

        return summary


# # =======================
# # 5. Usage Example
# # =======================
#
# if __name__ == "__main__":
#     from pymongo import MongoClient
#
#     # Kết nối MongoDB
#     client = MongoClient("mongodb://localhost:27017/")
#     db = client["interviewer_ai"]
#     collection = db["interview_results"]
#     interviewer = AdaptiveInterviewer()
#
#     # Test cases
#     test_cases = [
#         ("Ngô Văn Phát,KT1", "Kiểu dữ liệu trong Java"),
#
#     ]
#
