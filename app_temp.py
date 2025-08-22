import streamlit as st
import google.generativeai as genai
from PIL import Image
import io, os, re, json, time

# ====== Regex & chuẩn hoá ======
# Khớp các biến thể: "37-M1 56341", "37M156341", "37-M1 56.341" (cho phép seri như M1, D1)
FALLBACK_REGEX = re.compile(
    r"(\d{2})\s*[- ]?\s*([A-Z]{1,2}\d?)\s*[- ]?\s*(\d{4,6})"
)

def normalize_plate(raw: str) -> str:
    """
    Chuẩn hoá chuỗi biển số:
    - Bỏ ký tự lạ, giữ A-Z 0-9
    - Định dạng NN-SS DDD.DD hoặc NN-SS DDD.DDD (SS có thể chứa số, ví dụ: M1, D1)
    """
    s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    m = re.fullmatch(r"(\d{2})([A-Z]{1,2}\d?)([\d]{4,6})", s)
    if not m:
        return s
    p, series, num = m.groups()
    
    # Định dạng số cuối (4-6 chữ số thành DD.DD hoặc DDD.DD)
    if len(num) == 4:
        return f"{p}-{series} {num[:2]}.{num[2:]}"
    elif len(num) == 5:
        return f"{p}-{series} {num[:2]}.{num[2:]}"
    elif len(num) == 6:
        return f"{p}-{series} {num[:3]}.{num[3:]}"
    return f"{p}-{series} {num}"

def dedupe_preserve_order(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

# ====== Model wrapper ======
class GeminiModel:
    def __init__(self, api_key: str, model_name="gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_text_from_image(self, image_bytes: bytes, retries: int = 2):
        """
        Yêu cầu Gemini trả về JSON array, sau đó parse + chuẩn hoá.
        Có fallback: bóc từ text tự do bằng regex nếu mô hình không tuân thủ.
        """
        prompt = (
            "Extract ALL Vietnamese vehicle license plates in the image. "
            "Return ONLY a JSON array of uppercase strings with format NN-SS DDD.DD or NN-SS DDD.DDD. "
            "Example: ['37-M1 56.341','29-AY 005.40']. Do not add explanations."
        )
        for attempt in range(retries):
            try:
                img = Image.open(io.BytesIO(image_bytes))
                resp = self.model.generate_content([prompt, img])
                txt = (resp.text or "").strip()

                # Debug: Hiển thị đầu ra thô từ Gemini
                st.write("Đầu ra thô từ Gemini:", txt)

                # Gỡ code-fence nếu có ```json ... ```
                if txt.startswith("```"):
                    txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
                    txt = re.sub(r"\n?```$", "", txt).strip()

                # 1) Thử parse JSON đúng chuẩn
                plates = []
                try:
                    data = json.loads(txt)
                    if isinstance(data, list):
                        plates = [normalize_plate(str(x)) for x in data]
                        plates = [p for p in plates if re.fullmatch(r"\d{2}-[A-Z]{1,2}\d?\s+\d{2,3}\.\d{2,3}", p)]
                except Exception as e:
                    st.write("Lỗi parse JSON:", e)

                # 2) Fallback: lôi từ text tự do bằng regex
                if not plates:
                    U = txt.upper()
                    cand = [normalize_plate("".join(m.groups())) for m in FALLBACK_REGEX.finditer(U)]
                    plates = [p for p in cand if re.fullmatch(r"\d{2}-[A-Z]{1,2}\d?\s+\d{2,3}\.\d{2,3}", p)]

                return dedupe_preserve_order(plates)
            except Exception as e:
                st.error(f"Lỗi xử lý ({attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(1.5)
                else:
                    return []

# ====== Streamlit UI ======
def main():
    st.set_page_config(page_title="Nhận Diện Biển Số Xe", page_icon="🔎")
    st.title("Nhận Diện Biển Số Xe")

    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("GEMINI_API_KEY", type="password")
    if not api_key:
        st.info("Nhập GEMINI_API_KEY hoặc cấu hình biến môi trường khi deploy.")
        return

    try:
        gemini = GeminiModel(api_key)
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {e}")
        return

    up = st.file_uploader("Chọn ảnh (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"])
    if up:
        img_bytes = up.read()
        st.image(img_bytes, caption="Ảnh đầu vào", use_column_width=True)
        if st.button("Nhận Diện Biển Số"):
            with st.spinner("Đang xử lý..."):
                plates = gemini.extract_text_from_image(img_bytes)
            if plates:
                st.success("Kết quả nhận diện:")
                for p in plates: st.write(p)
                st.download_button("Tải xuống kết quả", "\n".join(plates),
                                   file_name="plates.txt", mime="text/plain")
            else:
                st.error("Không nhận diện được biển số phù hợp.")

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
