import json
import os

def normalize_arabic(text):
    """تبسيط النص العربي لتحسين مطابقة البحث (إزالة الهمزات والمسافات الزائدة)."""
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ة": "ه", "ى": "ي"
    }
    text = text.strip()
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def load_dataset(file_path):
    """تحميل ملف JSON والتأكد من ترميز UTF-8."""
    if not os.path.exists(file_path):
        print(f"خطأ: الملف غير موجود في {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    dataset_path = r"d:\البرمجيات - نسخ احتياطية\jameel-ai\data\dataset.json"
    data = load_dataset(dataset_path)
    
    # تحضير البيانات المطبوعة مسبقاً للسرعة
    processed_data = [
        {"prompt": normalize_arabic(item['prompt']), "response": item['response']}
        for item in data
    ]

    print("=== المساعد العربي جاهز! (اكتب 'خروج' للإنهاء) ===")
    
    while True:
        user_input = input("أنت: ")
        if user_input.lower() in ['خروج', 'exit', 'quit']:
            print("مع السلامة!")
            break
            
        query = normalize_arabic(user_input)
        # البحث عن تطابق
        match = next((item['response'] for item in processed_data if item['prompt'] == query), None)
        
        print(f"المساعد: {match if match else 'عذراً، لم أفهم ذلك حالياً.'}")

if __name__ == "__main__":
    main()