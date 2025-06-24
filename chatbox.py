import os
import sys
import re
import logging
from openai import OpenAI
import google.generativeai as genai

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """Loại bỏ ký tự đặc biệt nhưng giữ dấu tiếng Việt, dấu phẩy và dấu chấm"""
    cleaned = re.sub(r'[^\w\s\u00C0-\u1EF9,.]', '', text)
    cleaned = ' '.join(cleaned.split())
    return cleaned

def validate_api_key(api_key, provider):
    """Kiểm tra API key có hợp lệ về mặt định dạng"""
    if not api_key:
        raise ValueError("API key không được để trống")
    if len(api_key.strip()) != len(api_key):
        raise ValueError("API key chứa ký tự khoảng trắng không hợp lệ")
    
    # Kiểm tra định dạng cơ bản theo provider
    if provider == 'openai' and not api_key.startswith('sk-'):
        raise ValueError("OpenAI API key phải bắt đầu bằng 'sk-'")
    elif provider == 'gemini' and not api_key.startswith('AIza'):
        raise ValueError("Gemini API key thường bắt đầu bằng 'AIza'")
    
    return api_key

def generate_with_gemini(prompt, api_key, model_name="gemini-1.5-flash"):
    """Sinh kịch bản sử dụng Gemini"""
    try:
        # Kiểm tra model name hợp lệ
        if not model_name.startswith('gemini-'):
            model_name = "gemini-1.5-flash"  # Fallback to default
            logger.warning(f"Tên model không hợp lệ, sử dụng mặc định: {model_name}")

        # Kiểm tra API key
        logger.info("Kiểm tra API key cho Gemini")
        api_key = validate_api_key(api_key, 'gemini')
        logger.info("API key hợp lệ về định dạng")

        # Cấu hình Gemini
        logger.info(f"Cấu hình Gemini với model {model_name}")
        genai.configure(api_key=api_key)
        
        # Kiểm tra model có sẵn
        available_models = [m.name for m in genai.list_models()]
        if model_name not in available_models:
            logger.warning(f"Model {model_name} không khả dụng, sử dụng model mặc định")
            model_name = "gemini-1.5-flash"
        
        model = genai.GenerativeModel(model_name)

        # Tạo prompt
        mc_prompt = f"Viết kịch bản cho người dẫn chương trình bằng tiếng Việt, phong cách chuyên nghiệp, tự nhiên và hấp dẫn, chỉ chứa lời thoại trực tiếp, không bao gồm tiêu đề hoặc mô tả sự kiện. Nội dung dựa trên yêu cầu sau: {prompt}"

        # Gửi yêu cầu
        logger.info("Gửi yêu cầu đến Gemini")
        response = model.generate_content(
            mc_prompt,
            generation_config={
                "max_output_tokens": 150,
                "temperature": 0.7,
            }
        )
        
        # Kiểm tra phản hồi
        if not response.text:
            raise RuntimeError("Không nhận được phản hồi từ Gemini")
            
        logger.info("Nhận phản hồi thành công từ Gemini")
        return clean_text(response.text)
    except Exception as e:
        logger.error(f"Lỗi khi sử dụng Gemini: {str(e)}")
        raise RuntimeError(f"Lỗi Gemini: {str(e)}")

def generate_with_openai(prompt, api_key, model_name="gpt-4-turbo"):
    """Sinh kịch bản sử dụng OpenAI"""
    try:
        # Kiểm tra API key
        logger.info("Kiểm tra API key cho OpenAI")
        api_key = validate_api_key(api_key, 'openai')
        logger.info("API key hợp lệ về định dạng")

        # Cấu hình OpenAI
        client = OpenAI(api_key=api_key)
        
        mc_prompt = f"Viết kịch bản cho người dẫn chương trình bằng tiếng Việt, phong cách chuyên nghiệp, tự nhiên và hấp dẫn, chỉ chứa lời thoại trực tiếp, không bao gồm tiêu đề hoặc mô tả sự kiện. Nội dung dựa trên yêu cầu sau: {prompt}"
        
        # Gửi yêu cầu đến OpenAI
        logger.info(f"Gửi yêu cầu đến OpenAI với model {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là một người dẫn chương trình chuyên nghiệp."},
                {"role": "user", "content": mc_prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        logger.info("Nhận phản hồi thành công từ OpenAI")
        return clean_text(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Lỗi khi sử dụng OpenAI: {str(e)}")
        raise RuntimeError(f"Lỗi OpenAI: {str(e)}")

def main():
    if len(sys.argv) < 2:
        logger.error("Thiếu nội dung cần tạo")
        print("Vui lòng cung cấp nội dung cần tạo.")
        sys.exit(1)

    prompt = sys.argv[1]
    
    # Lấy cấu hình từ biến môi trường
    ai_provider = os.getenv('AI_PROVIDER', 'gemini').lower()
    api_key = os.getenv('API_KEY')
    model_name = os.getenv('MODEL_NAME', 'gemini-1.5-flash')

    if not api_key:
        logger.error("API key chưa được cấu hình")
        print("Lỗi: API key chưa được cấu hình")
        sys.exit(1)

    logger.info(f"Sử dụng AI provider: {ai_provider}, model: {model_name}")
    
    try:
        if ai_provider == 'gemini':
            try:
                generated_script = generate_with_gemini(prompt, api_key, model_name)
            except RuntimeError as e:
                logger.warning(f"Gemini thất bại: {str(e)}")
                # Chỉ thử chuyển sang OpenAI nếu API key có vẻ là của OpenAI
                if api_key.startswith('sk-'):
                    logger.info("Thử sử dụng OpenAI vì API key có định dạng OpenAI")
                    generated_script = generate_with_openai(prompt, api_key, model_name="gpt-4-turbo")
                else:
                    raise RuntimeError("Không thể sử dụng Gemini và không có API key OpenAI hợp lệ")
        else:
            generated_script = generate_with_openai(prompt, api_key, model_name)
        
        print(generated_script)
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tạo kịch bản: {str(e)}")
        print(f"Lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()