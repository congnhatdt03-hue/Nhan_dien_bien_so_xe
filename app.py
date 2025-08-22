# app.py — Streamlit + Gemini (ảnh), trả về JSON + chuẩn hoá biển số
# Phiên bản tối ưu: Cải thiện regex, thêm cache, tối ưu hiệu suất và bảo mật.

import streamlit as st
import google.generativeai as genai
from PIL import Image
import io, os, re, json, time, hashlib

# ====== Cấu hình và biến môi trường ======
DEFAULT_MODEL_NAME = "gemini-1.5-flash"  # Sử dụng model hợp lệ
DEFAULT_RETRIES = 2
DEFAULT_DEBUG_MODE = False  # Giá trị mặc định

# ====== Regex & chuẩn hoá ======
# Khớp các biến thể: "37-M1 56341", "37M156341", "37-M1 56.341" (cho phép seri như M1, D1 chứa số)
FALLBACK_REGEX = re.compile(
    r"(\d{2})\s*[- ]?\s*([A-Z]{1,2}\d?)\s*[- ]?\s*(\d{4,6})"
)

def normalize_plate(raw: str) -> str:
    """
    Chuẩn hoá chuỗi biển số:
    - Bỏ ký tự lạ, giữ A-Z 0-9
    - Định dạng NN-SS DD.DDD hoặc NN-SS DDD.DD (SS có thể chứa số, ví dụ: M1, D1)
    """
    s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    m = re.fullmatch(r"(\d{2})([A-Z]{1,2}\d?)(\d{4,6})", s)
    if not m:
        return s
    p, series, num = m.groups()
    # Tối ưu định dạng số cuối dựa trên độ dài
    if len(num) == 4:
        formatted_num = f"{num[:2]}.{num[2:]}"
    elif len(num) == 5:
        formatted_num = f"{num[:2]}.{num[3:]}"
    elif len(num) == 6:
        formatted_num = f"{num[:3]}.{num[3:]}"
    else:
        formatted_num = num
    return f"{p}-{series} {formatted_num}"

def dedupe_preserve_order(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ====== Model wrapper ======
class GeminiModel:
    def __init__(self, api_key: str, model_name=DEFAULT_MODEL_NAME):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    @st.cache_data(ttl=3600, show_spinner=False)  # Cache kết quả 1 giờ
    def extract_text_from_image(self, image_hash: str, image_bytes: bytes, debug_mode: bool = DEFAULT_DEBUG_MODE) -> list:
        """
        Yêu cầu Gemini trả về JSON array, sau đó parse + chuẩn hoá.
        Có fallback: bóc từ text tự do bằng regex nếu mô hình không tuân thủ.
        """
        prompt = (
            "Extract ALL Vietnamese vehicle license plates in the image. "
            "Return ONLY a JSON array of uppercase strings with format NN-SS DD.DDD or NN-SS DDD.DD. "
            "Example: ['37-M1 56.341','29-AY 005.40']. Do not add explanations."
        )
        for attempt in range(DEFAULT_RETRIES):
            try:
                img = Image.open(io.BytesIO(image_bytes))
                resp = self.model.generate_content([prompt, img])
                txt = (resp.text or "").strip()

                if debug_mode:
                    st.write("Đầu ra thô từ Gemini:", txt)

                # Gỡ code-fence nếu có ```json ... ```
                txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
                txt = re.sub(r"\n?```$", "", txt).strip()

                # 1) Thử parse JSON đúng chuẩn
                plates = []
                try:
                    data = json.loads(txt)
                    if isinstance(data, list):
                        plates = [normalize_plate(str(x)) for x in data]
                        plates = [p for p in plates if re.fullmatch(r"\d{2}-[A-Z]{1,2}\d?\s+\d{2,3}\.\d{2,3}", p)]
                except json.JSONDecodeError as e:
                    if debug_mode:
                        st.write("Lỗi parse JSON:", str(e))

                # 2) Fallback: lôi từ text tự do bằng regex
                if not plates:
                    U = txt.upper()
                    cand = [normalize_plate("".join(m.groups())) for m in FALLBACK_REGEX.finditer(U)]
                    plates = [p for p in cand if re.fullmatch(r"\d{2}-[A-Z]{1,2}\d?\s+\d{2,3}\.\d{2,3}", p)]

                return dedupe_preserve_order(plates)
            except Exception as e:
                if debug_mode:
                    st.error(f"Lỗi xử lý ({attempt + 1}/{DEFAULT_RETRIES}): {str(e)}")
                if attempt < DEFAULT_RETRIES - 1:
                    time.sleep(1.5)
                else:
                    return []
        return []

# ====== Streamlit UI ======
def main():
    st.set_page_config(page_title="Nhận Diện Biển Số Xe", page_icon="🔎", layout="wide")
    st.title("Nhận Diện Biển Số Xe")

    # Tối ưu bảo mật: Ẩn API key sau khi nhập
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.getenv("GEMINI_API_KEY") or ""
    api_key = st.session_state["api_key"]
    if not api_key:
        api_key = st.text_input("GEMINI_API_KEY", type="password")
        if api_key:
            st.session_state["api_key"] = api_key
    if not api_key:
        st.info("Nhập GEMINI_API_KEY hoặc cấu hình biến môi trường khi deploy.")
        return

    try:
        gemini = GeminiModel(api_key=st.session_state["api_key"])
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {str(e)}")
        return

    # Sidebar tùy chọn (tối ưu UX)
    with st.sidebar:
        st.header("Tùy chọn")
        debug_mode = st.checkbox("Bật chế độ debug", value=DEFAULT_DEBUG_MODE)

    up = st.file_uploader("Chọn ảnh (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"])
    if up:
        img_bytes = up.read()
        st.image(img_bytes, caption="Ảnh đầu vào", use_column_width=True)
        # Tạo hash ảnh để cache
        image_hash = hashlib.md5(img_bytes).hexdigest()
        if st.button("Nhận Diện Biển Số"):
            with st.spinner("Đang xử lý..."):
                plates = gemini.extract_text_from_image(image_hash, img_bytes, debug_mode)
            if plates:
                st.success("Kết quả nhận diện:")
                for p in plates:
                    st.write(p)
                st.download_button("Tải xuống kết quả", "\n".join(plates),
                                   file_name="plates.txt", mime="text/plain")
            else:
                st.error("Không nhận diện được biển số phù hợp. Hãy thử ảnh rõ nét hơn.")

    # Watermark
    st.markdown(
        """
        <div style='position: fixed; bottom: 10px; right: 10px; font-size: 14px; color: gray;'>
            Đồ án II, GVHD: ThS. Nguyễn Thị Huế
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
