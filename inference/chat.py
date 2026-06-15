import json
import gzip
import math
import re
import sys
import ast
import operator
import random
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.dialect import to_sudanese_text
from inference.learning import load_learned_examples

RIYADH = ZoneInfo("Asia/Riyadh")
ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
WEEKDAYS = [
    "الاثنين",
    "الثلاثاء",
    "الأربعاء",
    "الخميس",
    "الجمعة",
    "السبت",
    "الأحد",
]
GREGORIAN_MONTHS = [
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
]
HIJRI_MONTHS = [
    "محرم",
    "صفر",
    "ربيع الأول",
    "ربيع الآخر",
    "جمادى الأولى",
    "جمادى الآخرة",
    "رجب",
    "شعبان",
    "رمضان",
    "شوال",
    "ذو القعدة",
    "ذو الحجة",
]
UNKNOWN_RESPONSE = (
    "لا تتوفر لدي معلومات كافية للإجابة عن هذا السؤال حاليًا، "
    "ولم أتمكن من العثور على مصدر مناسب عبر الإنترنت."
)
UNKNOWN_RESPONSES = [
    UNKNOWN_RESPONSE,
    "لا أملك معلومة موثوقة عن هذا الموضوع حاليًا، كما لم أجد مصدرًا مناسبًا عبر الإنترنت.",
    "لم أتمكن من الوصول إلى إجابة موثوقة لهذا السؤال الآن.",
]
RESULT_TEMPLATES = [
    "الناتج هو {result}.",
    "الإجابة: {result}.",
    "يساوي {result}.",
    "بعد الحساب، النتيجة {result}.",
]
DIVISION_BY_ZERO_RESPONSES = [
    "لا يمكن القسمة على صفر.",
    "هذه العملية غير ممكنة لأن المقسوم عليه يساوي صفرًا.",
    "القسمة على صفر غير معرّفة رياضيًا.",
]
DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬",
    "01234567890123456789.,",
)
ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
ARABIC_NUMBER_WORDS = {
    "صفر": "0",
    "واحد": "1",
    "واحدة": "1",
    "اثنان": "2",
    "اثنين": "2",
    "اثنتان": "2",
    "ثلاثة": "3",
    "اربعة": "4",
    "أربعة": "4",
    "خمسة": "5",
    "ستة": "6",
    "سبعة": "7",
    "ثمانية": "8",
    "تسعة": "9",
    "عشرة": "10",
}
ARABIC_STOP_WORDS = {
    "ما",
    "ماذا",
    "من",
    "هو",
    "هي",
    "هل",
    "في",
    "عن",
    "على",
    "إلى",
    "الى",
    "متى",
    "أين",
    "اين",
    "كيف",
    "كم",
    "لماذا",
    "وما",
    "وهو",
    "وهي",
    "ولد",
    "ولدت",
    "تقع",
    "يقع",
    "حدث",
    "حدثت",
    "تأسس",
    "تأسست",
}


def arabic_number(value):
    return str(value).translate(ARABIC_DIGITS)


def format_number(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        text = f"{value:.10f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.translate(ARABIC_DIGITS)


def format_grouped_number(value):
    if isinstance(value, float) and not value.is_integer():
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{int(value):,}"
    return text.translate(ARABIC_DIGITS)


def varied_result(value):
    return random.choice(RESULT_TEMPLATES).format(result=format_number(value))


def evaluate_math_node(node):
    if isinstance(node, ast.Expression):
        return evaluate_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
        left = evaluate_math_node(node.left)
        right = evaluate_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 20:
            raise ValueError("Exponent is too large")
        return ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
        return ALLOWED_UNARY_OPERATORS[type(node.op)](evaluate_math_node(node.operand))
    raise ValueError("Unsupported mathematical expression")


def safe_calculate(expression):
    if len(expression) > 120:
        raise ValueError("Expression is too long")
    parsed = ast.parse(expression, mode="eval")
    return evaluate_math_node(parsed)


def normalize_math_expression(text):
    normalized = text.translate(DIGIT_TRANSLATION).lower()
    replacements = (
        ("مضروبا في", "*"),
        ("مضروب في", "*"),
        ("مضروبة في", "*"),
        ("مقسوم على", "/"),
        ("قسمة", "/"),
        ("÷", "/"),
        ("×", "*"),
        ("ضرب", "*"),
        ("زائد", "+"),
        ("جمع", "+"),
        ("ناقص", "-"),
        ("طرح", "-"),
        ("أس", "**"),
        ("^", "**"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    return normalized


def math_response(prompt):
    verbal_text = prompt.translate(DIGIT_TRANSLATION).lower()
    for number_word, number_value in ARABIC_NUMBER_WORDS.items():
        optional_conjunction = "" if number_word.startswith("و") else "(?:و)?"
        verbal_text = re.sub(
            rf"\b{optional_conjunction}{re.escape(number_word)}\b",
            f" {number_value} ",
            verbal_text,
        )
    plain_verbal_text = re.sub(r"[\u064b-\u065f\u0670]", "", verbal_text)
    normalized = normalize_math_expression(prompt)

    if any(word in plain_verbal_text for word in ("اقسم", "قسم")):
        division_numbers = re.findall(r"-?\d+(?:\.\d+)?", plain_verbal_text)
        if len(division_numbers) >= 2 and "على" in plain_verbal_text:
            first, second = map(float, division_numbers[:2])
            if second == 0:
                return random.choice(DIVISION_BY_ZERO_RESPONSES)
            return varied_result(first / second)

    verbal_operations = (
        (
            r"(?:اجمع|مجموع)\s+(-?\d+(?:\.\d+)?)\s+(?:(?:و|مع)\s+)?"
            r"(-?\d+(?:\.\d+)?)",
            "+",
        ),
        (
            r"(?:اطرح|طرح)\s+(-?\d+(?:\.\d+)?)\s+من\s+"
            r"(-?\d+(?:\.\d+)?)",
            "reverse_subtract",
        ),
        (r"(?:فرق)\s+(-?\d+(?:\.\d+)?)\s+(?:و|عن)\s*(-?\d+(?:\.\d+)?)", "-"),
        (r"(?:اضرب|حاصل ضرب)\s+(-?\d+(?:\.\d+)?)\s+(?:في|ب)\s*(-?\d+(?:\.\d+)?)", "*"),
        (
            r"(?:ا?قسم|قسّم)\s+(-?\d+(?:\.\d+)?)\s+على\s+(-?\d+(?:\.\d+)?)",
            "/",
        ),
    )
    for pattern, symbol in verbal_operations:
        operation_match = re.search(pattern, verbal_text)
        if not operation_match:
            continue
        first = float(operation_match.group(1))
        second = float(operation_match.group(2))
        if symbol == "reverse_subtract":
            expression = f"{second}-{first}"
        else:
            expression = f"{first}{symbol}{second}"
        try:
            result = safe_calculate(expression)
        except ZeroDivisionError:
            return random.choice(DIVISION_BY_ZERO_RESPONSES)
        return varied_result(result)

    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:من)?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if not percentage_match:
        percentage_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:بالمئة|بالمائة|في المئة|في المائة)\s*من\s*"
            r"(\d+(?:\.\d+)?)",
            normalized,
        )
    if percentage_match:
        percentage = float(percentage_match.group(1))
        number = float(percentage_match.group(2))
        result = percentage * number / 100
        return (
            f"{format_number(percentage)}٪ من {format_number(number)} "
            f"تساوي {format_number(result)}."
        )

    square_root_match = re.search(
        r"(?:الجذر التربيعي|جذر)\s*(?:ل(?:ل|ـ)?|من)?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if square_root_match:
        number = float(square_root_match.group(1))
        result = math.sqrt(number)
        return f"الجذر التربيعي لـ {format_number(number)} يساوي {format_number(result)}."

    expression_candidates = re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:\s*(?:\*\*|[+\-*/%])\s*[-+]?\d+(?:\.\d+)?)+",
        normalized,
    )
    if not expression_candidates:
        return None

    expression = max(expression_candidates, key=len).replace(" ", "")
    try:
        result = safe_calculate(expression)
    except ZeroDivisionError:
        return random.choice(DIVISION_BY_ZERO_RESPONSES)
    except (SyntaxError, ValueError, OverflowError):
        return "لم أتمكن من تحليل العملية الرياضية. اكتبها بصيغة أوضح."

    if isinstance(result, complex) or not math.isfinite(float(result)):
        return "نتيجة العملية غير صالحة ضمن الأعداد الحقيقية."
    return varied_result(result)


def gregorian_to_julian_day(year, month, day):
    adjustment = (14 - month) // 12
    adjusted_year = year + 4800 - adjustment
    adjusted_month = month + 12 * adjustment - 3
    return (
        day
        + (153 * adjusted_month + 2) // 5
        + 365 * adjusted_year
        + adjusted_year // 4
        - adjusted_year // 100
        + adjusted_year // 400
        - 32045
    )


def islamic_to_julian_day(year, month, day):
    return (
        day
        + math.ceil(29.5 * (month - 1))
        + (year - 1) * 354
        + math.floor((3 + 11 * year) / 30)
        + 1948439
        - 1
    )


def gregorian_to_hijri(year, month, day):
    julian_day = gregorian_to_julian_day(year, month, day)
    hijri_year = math.floor((30 * (julian_day - 1948439) + 10646) / 10631)
    hijri_month = min(
        12,
        math.ceil(
            (julian_day - 29 - islamic_to_julian_day(hijri_year, 1, 1)) / 29.5
        )
        + 1,
    )
    hijri_month = max(1, hijri_month)
    hijri_day = (
        julian_day - islamic_to_julian_day(hijri_year, hijri_month, 1) + 1
    )
    return hijri_year, hijri_month, hijri_day


def temporal_response(prompt):
    normalized = prompt.strip().lower().replace("؟", "")
    asks_time = any(
        word in normalized
        for word in ("الساعة", "الساعه", "الوقت", "كم الساعة", "كم الساعه")
    )
    asks_day = any(
        word in normalized
        for word in ("اليوم ايش", "ما اليوم", "وش اليوم", "اي يوم", "اسم اليوم")
    )
    asks_date = any(
        word in normalized for word in ("التاريخ", "تاريخ اليوم", "كم التاريخ")
    )
    asks_hijri = "هجري" in normalized
    asks_gregorian = "ميلادي" in normalized

    if not any((asks_time, asks_day, asks_date, asks_hijri, asks_gregorian)):
        return None

    now = datetime.now(RIYADH)
    hijri_year, hijri_month, hijri_day = gregorian_to_hijri(
        now.year, now.month, now.day
    )
    parts = []

    if asks_day or asks_date:
        parts.append(f"اليوم {WEEKDAYS[now.weekday()]}")
    if asks_gregorian or (asks_date and not asks_hijri):
        parts.append(
            f"التاريخ الميلادي {arabic_number(now.day)} "
            f"{GREGORIAN_MONTHS[now.month - 1]} {arabic_number(now.year)}"
        )
    if asks_hijri or (asks_date and not asks_gregorian):
        parts.append(
            f"التاريخ الهجري التقريبي {arabic_number(hijri_day)} "
            f"{HIJRI_MONTHS[hijri_month - 1]} {arabic_number(hijri_year)} هـ"
        )
    if asks_time:
        clock = now.strftime("%I:%M").lstrip("0").translate(ARABIC_DIGITS)
        period = "صباحًا" if now.hour < 12 else "مساءً"
        parts.append(f"الساعة {clock} {period}")

    return "، و".join(parts) + " بتوقيت الرياض."


def load_model():
    import torch

    from models.model import GreetingLanguageModel

    checkpoint = torch.load(ROOT / "models" / "greeting_model.pt", map_location="cpu")
    config = checkpoint["config"]
    model = GreetingLanguageModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["char_to_id"]


def retrieve_response(prompt):
    with (ROOT / "data" / "dataset.json").open(encoding="utf-8") as file:
        examples = json.load(file)
    examples.extend(load_learned_examples())

    normalized = prompt.strip().lower()
    scored_examples = [
        (
            SequenceMatcher(
                None, normalized, item["prompt"].strip().lower()
            ).ratio(),
            item,
        )
        for item in examples
    ]
    score, best = max(scored_examples, key=lambda result: result[0])
    if score < 0.62:
        return None
    greeting_words = {
        "مرحبا",
        "مرحباً",
        "أهلا",
        "اهلا",
        "هلا",
        "السلام",
        "صباح",
        "مساء",
        "هاي",
        "hello",
    }
    if any(word in normalized for word in greeting_words):
        greeting_examples = [
            item
            for item in examples
            if any(
                word in item["prompt"].strip().lower()
                for word in greeting_words
            )
        ]
        if greeting_examples:
            close_greetings = [
                item
                for item in greeting_examples
                if SequenceMatcher(
                    None, normalized, item["prompt"].strip().lower()
                ).ratio()
                >= max(0.55, score - 0.2)
            ]
            if close_greetings:
                return random.choice(close_greetings)["response"]
    return best["response"]


def normalize_words(text):
    cleaned = re.sub(r"[^\w\u0600-\u06FF]+", " ", text.lower())
    return {
        word
        for word in cleaned.split()
        if len(word) > 1 and word not in ARABIC_STOP_WORDS
    }


def internet_search_terms(question):
    words = re.sub(r"[^\w\u0600-\u06FF]+", " ", question.lower()).split()
    useful_words = [
        word
        for word in words
        if word not in ARABIC_STOP_WORDS and len(word) > 1
    ]
    return " ".join(useful_words) or question


QUESTION_STARTERS = (
    "ما",
    "ماذا",
    "من",
    "متى",
    "أين",
    "اين",
    "كيف",
    "كم",
    "هل",
    "لماذا",
    "احسب",
    "اجمع",
    "اطرح",
    "اقسم",
)

ARABIC_COUNTS = {
    "واحد": 1,
    "واحدة": 1,
    "اثنان": 2,
    "اثنين": 2,
    "ثلاث": 3,
    "ثلاثة": 3,
    "أربع": 4,
    "اربعة": 4,
    "أربعة": 4,
    "خمس": 5,
    "خمسة": 5,
    "عشر": 10,
    "عشرة": 10,
}
CONCEPT_CACHE = {}
LOCAL_DEFINITIONS = {
    "طاقة الشمسية": (
        "الطاقة الشمسية طاقة متجددة جاية من ضوء وحرارة الشمس، "
        "وبتتحول لكهرباء باستخدام الألواح الشمسية أو بتستخدم للتسخين."
    ),
    "ذكاء الاصطناعي": (
        "الذكاء الاصطناعي تقنيات بتخلي الحاسوب ينفذ مهام محتاجة عادةً "
        "ذكاء بشري، زي الفهم والتعلّم والتحليل واتخاذ القرار."
    ),
    "تعلم الآلة": (
        "تعلّم الآلة فرع من الذكاء الاصطناعي بخلي الأنظمة تتعلّم الأنماط "
        "من البيانات عشان تتنبأ أو تتخذ قرارات."
    ),
    "انترنت": (
        "الإنترنت شبكة عالمية بتربط أجهزة وشبكات كتيرة عشان تتبادل "
        "المعلومات والخدمات."
    ),
    "حاسوب": (
        "الحاسوب جهاز إلكتروني بستقبل البيانات وبعالجها حسب برامج وتعليمات "
        "وبطلع نتائج قابلة للاستخدام."
    ),
    "جاذبية": (
        "الجاذبية قوة طبيعية بتجذب الأجسام البعندها كتلة لبعض، وهي البتخلي "
        "الأجسام تقع ناحية الأرض."
    ),
    "ذرة": (
        "الذرة أصغر وحدة من العنصر بتحافظ على خواصه الكيميائية، وبتتكون من "
        "نواة وإلكترونات حولها."
    ),
    "تمثيل ضوئي": (
        "التمثيل الضوئي عملية بتحول فيها النباتات الضوء لطاقة كيميائية، "
        "باستخدام الماء وثاني أكسيد الكربون، وبتطلق الأكسجين."
    ),
    "طقس": (
        "الطقس هو حالة الجو في مكان وزمن محددين، زي الحرارة والرياح "
        "والأمطار والرطوبة."
    ),
    "مناخ": (
        "المناخ هو النمط المعتاد للطقس في منطقة معينة خلال فترة زمنية طويلة."
    ),
}
TRUSTED_CONCEPTS = {
    "حوسبة سحابية": {
        "title": "الحوسبة السحابية",
        "summary": (
            "الحوسبة السحابية نموذج بيوفّر موارد حوسبة زي الخوادم والتخزين "
            "والتطبيقات عبر الشبكة عند الطلب، مع إمكانية التوسّع والدفع حسب "
            "الاستخدام."
        ),
        "source": "https://csrc.nist.gov/pubs/sp/800/145/final",
    },
    "حوسبة السحابية": {
        "title": "الحوسبة السحابية",
        "summary": (
            "الحوسبة السحابية نموذج بيوفّر موارد حوسبة زي الخوادم والتخزين "
            "والتطبيقات عبر الشبكة عند الطلب، مع إمكانية التوسّع والدفع حسب "
            "الاستخدام."
        ),
        "source": "https://csrc.nist.gov/pubs/sp/800/145/final",
    },
    "طقس": {
        "title": "الطقس",
        "summary": (
            "الطقس هو حالة الغلاف الجوي في مكان وزمن محددين، زي الحرارة "
            "والرياح والأمطار، وبيتغير خلال ساعات أو أيام."
        ),
        "source": "https://wmo.int/topics/weather",
    },
    "مناخ": {
        "title": "المناخ",
        "summary": (
            "المناخ هو نمط ومتوسط أحوال الطقس في منطقة خلال مدة طويلة، "
            "وعادةً بتحلله البيانات الممتدة لعقود."
        ),
        "source": "https://wmo.int/topics/climate",
    },
    "بايثون": {
        "title": "بايثون",
        "summary": (
            "بايثون لغة برمجة عالية المستوى معروفة بوضوح صياغتها وسهولة "
            "قراءتها، وبتستخدم في التعليم والأتمتة والويب وتحليل البيانات."
        ),
        "source": "https://docs.python.org/3/tutorial/",
    },
    "جافا": {
        "title": "جافا",
        "summary": (
            "جافا لغة برمجة كائنية ومترجمة إلى تعليمات تعمل على آلة جافا "
            "الافتراضية، وبتستخدم كتير في الأنظمة المؤسسية والتطبيقات الكبيرة."
        ),
        "source": "https://docs.oracle.com/javase/tutorial/",
    },
}
DOMAIN_SOURCE_NAMES = {
    "dictionary": "مصدر معجمي",
    "medical": "مصدر طبي أو صحي",
    "technology": "توثيق تقني أو مرجع حوسبي",
    "economy": "مصدر اقتصادي أو إحصائي",
    "science": "مرجع علمي",
    "geography": "مصدر جغرافي أو إحصائي",
    "general": "مرجع عام",
}
SCIENTIFIC_TERM_TRANSLATIONS = {
    "تأثير": "effect",
    "اثر": "effect",
    "أثر": "effect",
    "النوم": "sleep",
    "نوم": "sleep",
    "الذاكرة": "memory",
    "ذاكرة": "memory",
    "التعلم": "learning",
    "تعلم": "learning",
    "الدماغ": "brain",
    "دماغ": "brain",
    "السرطان": "cancer",
    "سرطان": "cancer",
    "السكري": "diabetes",
    "سكري": "diabetes",
    "القلب": "heart",
    "المناخ": "climate",
    "مناخ": "climate",
    "التغير": "change",
    "الاحتباس": "warming",
    "الذكاء": "intelligence",
    "الاصطناعي": "artificial",
    "الطاقة": "energy",
    "الشمسية": "solar",
    "التعليم": "education",
    "الأطفال": "children",
    "الاطفال": "children",
    "القلق": "anxiety",
    "الاكتئاب": "depression",
    "الرياضة": "exercise",
    "التغذية": "nutrition",
}


def normalized_dialogue_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[؟?!.،؛;:]", " ", text.lower())).strip()


def dialogue_control_response(prompt):
    normalized = normalized_dialogue_text(prompt)

    if any(
        phrase in normalized
        for phrase in (
            "المعلومة طويلة",
            "الاجابة طويلة",
            "الإجابة طويلة",
            "الرد طويل",
            "عرضت معلومة طويلة",
            "كلام كتير",
            "طولت شديد",
            "طولت جدا",
        )
    ):
        return (
            "معاك حق، الرد كان طويل زيادة. من هسع حأديك الخلاصة المطلوبة "
            "مباشرة، وما بزيد التفاصيل إلا لو طلبتها."
        )

    if re.match(
        r"^(?:اختصر|اختصرها|اختصر لي|عاوزها مختصرة|عايزها مختصرة|"
        r"باختصار|الخلاصة بس|مختصر)",
        normalized,
    ):
        return (
            "تمام، حأخلي الإجابات مختصرة ومباشرة، وأعرض أهم نقطة بس."
        )

    if any(
        phrase in normalized
        for phrase in (
            "فصل اكتر",
            "فصّل أكتر",
            "فصل أكثر",
            "اشرح اكتر",
            "اشرح أكثر",
            "عاوز تفاصيل",
            "عايز تفاصيل",
            "بالتفصيل",
        )
    ):
        return "تمام، حأشرح بتفصيل أكتر مع ترتيب النقاط المهمة."

    if normalized in {
        "شكرا",
        "شكراً",
        "مشكور",
        "تسلم",
        "تمام شكرا",
        "كويس",
        "ممتاز",
    }:
        return "العفو، تحت أمرك."

    if any(
        phrase in normalized
        for phrase in (
            "الاجابة غلط",
            "الإجابة غلط",
            "الرد غلط",
            "المعلومة غلط",
            "دا خطأ",
            "ده خطأ",
            "ما صحيح",
            "غير صحيح",
        )
    ):
        return (
            "معاك حق إنك تنبّهني. ورّيني الجزء الغلط أو اكتب التصحيح، "
            "وأنا أراجع السؤال والمصدر وأديك إجابة مصححة."
        )

    if normalized in {"اعد الصياغة", "أعد الصياغة", "صيغها تاني", "قولها بطريقة تانية"}:
        return "أكيد. أرسل لي النص أو حدّد الإجابة العاوزني أصيغها تاني."

    return None


def last_assistant_answer(history):
    for item in reversed(history or []):
        if item.get("role") == "assistant" and item.get("content", "").strip():
            return item["content"].strip()
    return None


def dialogue_followup_response(prompt, history):
    normalized = normalized_dialogue_text(prompt)
    previous_answer = last_assistant_answer(history)
    if not previous_answer:
        return None

    if re.match(
        r"^(?:اختصر|اختصرها|اختصر لي|أديني الخلاصة|اديني الخلاصة|"
        r"الخلاصة بس|قولها باختصار|باختصار)",
        normalized,
    ):
        concise = shorten_answer(previous_answer, max_chars=260)
        concise = re.sub(
            r"^(?:عن|بالنسبة لـ|بالنسبة إلى)\s+[^:\n]{1,100}[،,:]\s*"
            r"(?:الإجابة|فالإجابة هي)?\s*:?\s*",
            "",
            concise,
        )
        return f"الخلاصة: {concise.strip()}"

    if normalized in {
        "اعدها",
        "أعدها",
        "كررها",
        "قولها تاني",
        "أعد الإجابة",
        "اعد الاجابة",
    }:
        return previous_answer

    return None


def response_detail_preference(prompt, history):
    messages = [
        item.get("content", "")
        for item in (history or [])
        if item.get("role") == "user"
    ]
    messages.append(prompt)
    for message in reversed(messages[-8:]):
        normalized = normalized_dialogue_text(message)
        if any(
            phrase in normalized
            for phrase in (
                "المعلومة طويلة",
                "الاجابة طويلة",
                "الإجابة طويلة",
                "الرد طويل",
                "كلام كتير",
                "اختصر",
                "باختصار",
                "الخلاصة بس",
                "مختصر",
            )
        ):
            return "brief"
        if any(
            phrase in normalized
            for phrase in (
                "فصل اكتر",
                "فصّل أكتر",
                "فصل أكثر",
                "اشرح اكتر",
                "اشرح أكثر",
                "عاوز تفاصيل",
                "عايز تفاصيل",
                "بالتفصيل",
            )
        ):
            return "detailed"
    return None


def shorten_answer(answer, max_chars=340):
    if len(answer) <= max_chars:
        return answer
    body, separator, sources = answer.partition("\n\nالمصدر:")
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!؟])\s+|\n+", body)
        if item.strip()
    ]
    selected = []
    total = 0
    for sentence in sentences:
        if len(sentence) > max_chars and not selected:
            clauses = [
                clause.strip()
                for clause in re.split(r"[،؛;]", sentence)
                if clause.strip()
            ]
            for clause in clauses:
                addition = len(clause) + (2 if selected else 0)
                if total + addition > max_chars and selected:
                    break
                selected.append(clause)
                total += addition
            break
        if total + len(sentence) > max_chars and selected:
            break
        selected.append(sentence)
        total += len(sentence)
        if len(selected) >= 3:
            break
    concise = " ".join(selected).strip()
    if separator:
        concise += f"\n\nالمصدر:{sources}"
    return concise


def apply_response_preference(answer, preference):
    if preference == "brief":
        return shorten_answer(answer, max_chars=260)
    if preference == "balanced":
        return shorten_answer(answer, max_chars=700)
    return answer


def compact_scientific_answer(answer, max_studies):
    body, separator, sources = answer.partition("\n\nالمصدر:")
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return answer
    selected = [lines[0]]
    studies = 0
    include_current = False
    consensus = None
    for line in lines[1:]:
        if re.match(r"^\d+\.", line):
            studies += 1
            include_current = studies <= max_studies
        if line.startswith("الخلاصة العلمية:"):
            consensus = line
            include_current = False
            continue
        if include_current:
            selected.append(line)
    if consensus:
        selected.append(consensus)
    compact = "\n".join(selected)
    if separator:
        compact += f"\n\nالمصدر:{sources}"
    return compact


def apply_question_response_preference(question, answer, preference):
    if scientific_search_request(question):
        if preference == "brief":
            return compact_scientific_answer(answer, max_studies=1)
        if preference == "balanced":
            return compact_scientific_answer(answer, max_studies=2)
        return answer
    return apply_response_preference(answer, preference)


def split_questions(prompt):
    cleaned = " ".join(prompt.strip().split())
    if not cleaned:
        return []

    parts = [
        part.strip(" ،؛;؟?")
        for part in re.split(r"[؟?؛;\n]+", cleaned)
        if part.strip(" ،؛;؟?")
    ]
    expanded = []
    starter_pattern = "|".join(map(re.escape, QUESTION_STARTERS))
    for part in parts:
        subparts = re.split(
            rf"\s+(?=و(?:{starter_pattern})\b)",
            part,
            flags=re.IGNORECASE,
        )
        for item in subparts:
            item = item.strip(" ،")
            item = re.sub(
                rf"^و(?=(?:{starter_pattern})\b)",
                "",
                item,
                flags=re.IGNORECASE,
            )
            if item:
                expanded.append(item)

    return expanded[:6]


def question_domain(question):
    normalized = question.lower()
    domain_keywords = (
        ("dictionary", ("معنى", "مرادف", "ضد كلمة", "تعريف كلمة", "ترجمة كلمة")),
        ("medical", ("مرض", "دواء", "علاج", "أعراض", "طبي", "صحة", "تشخيص")),
        ("technology", ("برمجة", "برنامج", "تقنية", "حاسوب", "خوارزم", "بايثون", "جافا")),
        ("economy", ("اقتصاد", "ناتج محلي", "تضخم", "بطالة", "سكان", "دخل")),
        (
            "science",
            (
                "علم",
                "فيزياء",
                "كيمياء",
                "أحياء",
                "فضاء",
                "طاقة",
                "طقس",
                "مناخ",
            ),
        ),
        ("geography", ("دولة", "مدينة", "عاصمة", "سكان", "مساحة", "عملة", "لغة رسمية")),
    )
    for domain, keywords in domain_keywords:
        if any(keyword in normalized for keyword in keywords):
            return domain
    return "general"


SUDANESE_WELCOME_PHRASES = (
    "حبابك عشرة!",
    "حبابك مليون!",
    "حبابك طن!",
    "مشتاقين!",
)


def sudanese_greeting_response(normalized):
    welcome = random.choice(SUDANESE_WELCOME_PHRASES)
    if normalized.startswith("السلام عليكم"):
        return f"وعليكم السلام ورحمة الله وبركاته. {welcome}"
    return welcome


def exact_local_response(prompt):
    normalized = re.sub(r"[؟?!.،]", "", prompt.strip().lower())
    greeting_phrases = {
        "هلا بيك",
        "هلا والله",
        "مرحبا",
        "مرحبًا",
        "السلام عليكم",
        "السلام عليكم ورحمة الله",
        "أهلا",
        "اهلا",
        "أهلًا",
        "صباح الخير",
        "مساء الخير",
        "هاي",
        "hello",
    }
    greeting_phrases.update(
        {
            "هلا",
            "هلا والله",
            "مرحبا",
            "مرحباً",
            "مرحبًا",
            "يا مرحبا",
            "أهلا",
            "أهلاً",
            "اهلا",
            "أهلين",
            "مرحبتين",
            "حبابك",
            "مشتاقين",
            "السلام عليكم",
            "السلام عليكم ورحمة الله",
            "صباح الخير",
            "مساء الخير",
            "هاي",
        }
    )
    if normalized in greeting_phrases:
        return sudanese_greeting_response(normalized)
    return None


def open_request_response(prompt):
    normalized = re.sub(r"[؟?!.،]", "", prompt.strip().lower())
    if re.match(
        r"^(?:أعطني|اعطني|قدم لي|قدّم لي)\s+(?:بعض\s+)?نصائح(?:\s+عامة)?$",
        normalized,
    ):
        return (
            "بالتأكيد. إليك نصائح عامة مفيدة:\n"
            "1. حدّد أولوياتك اليومية ولا تشتت نفسك بين مهام كثيرة.\n"
            "2. خصص وقتًا منتظمًا للنوم والحركة والراحة.\n"
            "3. تعلّم شيئًا صغيرًا كل يوم وطبّقه عمليًا.\n"
            "4. راجع المعلومات المهمة من مصادر موثوقة.\n"
            "5. عند اتخاذ قرار كبير، قارن البدائل والنتائج المتوقعة."
        )
    if any(
        re.match(rf"^(?:أعطني|اعطني|قدم لي|قدّم لي)\s+{item}\b", normalized)
        for item in ("نصائح", "نصيحة", "اقتراحات")
    ):
        return "بالتأكيد. في أي مجال تريد النصائح تحديدًا؟"
    return None


def wikidata_entity_id(subject):
    params = urlencode(
        {
            "action": "wbsearchentities",
            "search": subject,
            "language": "ar",
            "uselang": "ar",
            "type": "item",
            "limit": "1",
            "format": "json",
        }
    )
    try:
        results = fetch_json(
            f"https://www.wikidata.org/w/api.php?{params}"
        ).get("search", [])
        return results[0]["id"] if results else None
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def fetch_json(url, attempts=1, timeout=6):
    last_error = None
    for attempt in range(attempts):
        request = Request(
            url,
            headers={"User-Agent": "ArabicAssistant/1.0 (local educational project)"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                delay = (
                    1.2
                    if isinstance(error, HTTPError) and error.code == 429
                    else 0.35
                )
                time.sleep(delay)
    raise last_error


def fetch_text(url):
    request = Request(
        url,
        headers={"User-Agent": "ArabicAssistant/1.0 (local educational project)"},
    )
    with urlopen(request, timeout=6) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            payload = gzip.decompress(payload)
        return payload.decode("utf-8", errors="replace")


def latest_python_answer(question):
    normalized = question.lower()
    if "بايثون" not in normalized or not any(
        word in normalized for word in ("أحدث", "احدث", "آخر إصدار", "اخر اصدار")
    ):
        return None
    try:
        page = fetch_text("https://www.python.org/downloads/")
        match = re.search(r"Download Python (\d+\.\d+(?:\.\d+)?)", page)
        if not match:
            return None
        return (
            f"أحدث إصدار مستقر معروض هو بايثون {match.group(1)}."
            "\n\nالمصدر: https://www.python.org/downloads/"
            "\nمرجع البيانات: الموقع الرسمي للغة بايثون"
        )
    except (HTTPError, URLError, TimeoutError, OSError):
        return None


def ranked_cities_request(question):
    normalized = question.lower().strip(" ؟?")
    if "مدن" not in normalized or not any(
        word in normalized for word in ("أكبر", "اكبر", "أكثر", "اكثر")
    ):
        return None

    count_match = re.search(r"\b(\d+)\b", normalized.translate(DIGIT_TRANSLATION))
    count = int(count_match.group(1)) if count_match else None
    if count is None:
        count = next(
            (value for word, value in ARABIC_COUNTS.items() if word in normalized),
            3,
        )
    count = max(1, min(count, 10))

    country_match = re.search(
        r"(?:في|بـ|داخل)\s+([\u0600-\u06FF][\u0600-\u06FF\s-]+)$",
        normalized,
    )
    if not country_match:
        return None
    country = re.split(
        r"\s+(?:من\s+حيث|من\s+ناحية|حسب|بناءً?\s+على)\s+",
        country_match.group(1).strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    metric = "area" if "مساحة" in normalized else "population"
    return count, country, metric


def ranked_cities_answer(question):
    request = ranked_cities_request(question)
    if not request:
        return None
    count, country, metric = request
    country_id = wikidata_entity_id(country)
    if not country_id:
        return None

    property_id = "P2046" if metric == "area" else "P1082"
    population_requirement = (
        "wdt:P1082 ?population;" if metric == "area" else ""
    )
    query = f"""
    SELECT ?city ?cityLabel (MAX(?value) AS ?rankValue) WHERE {{
      ?city wdt:P17 wd:{country_id};
            wdt:P31/wdt:P279* wd:Q515;
            {population_requirement}
            wdt:{property_id} ?value.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ar,en". }}
    }}
    GROUP BY ?city ?cityLabel
    ORDER BY DESC(?rankValue)
    LIMIT {count * 5}
    """
    url = "https://query.wikidata.org/sparql?" + urlencode(
        {"query": query, "format": "json"}
    )
    try:
        bindings = fetch_json(url, attempts=2, timeout=18).get(
            "results", {}
        ).get("bindings", [])
        if not bindings:
            return None
        unique_cities = []
        seen_labels = set()
        for item in bindings:
            city = item["cityLabel"]["value"]
            if metric == "area" and re.search(
                r"\b(?:مشروع|واحة|محافظة|منطقة|إقليم)\b",
                city,
                flags=re.IGNORECASE,
            ):
                continue
            normalized_city = city.strip().casefold()
            if normalized_city in seen_labels:
                continue
            seen_labels.add(normalized_city)
            unique_cities.append(item)
            if len(unique_cities) == count:
                break
        if len(unique_cities) < count:
            return None

        lines = []
        for index, item in enumerate(unique_cities, start=1):
            city = item["cityLabel"]["value"]
            value = float(item["rankValue"]["value"])
            if metric == "area":
                detail = f"{format_grouped_number(value)} كيلومترًا مربعًا"
            else:
                detail = f"{format_grouped_number(value)} نسمة"
            lines.append(f"{index}. {city}: {detail}.")
        criterion = "المساحة" if metric == "area" else "عدد السكان"
        return (
            f"أكبر {count} مدن في {country} حسب {criterion}:\n"
            + "\n".join(lines)
            + "\n\nالمصدر: https://query.wikidata.org/"
            + "\nمرجع البيانات: ويكي بيانات"
        )
    except (
        HTTPError,
        URLError,
        TimeoutError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ):
        return None


def comparison_request(question):
    cleaned = question.strip(" ؟?")
    cleaned = re.sub(
        r"\s+(?:من حيث|من ناحية|في|بالنسبة لـ?|لـ)\s+"
        r"[\u0600-\u06FF\w\s]+$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    patterns = (
        r"هل\s+(.+?)\s+أفضل\s+من\s+(.+)$",
        r"هل\s+(.+?)\s+(?:أحسن|احسن|أسهل|اسهل|أسرع|اسرع|"
        r"أقوى|اقوى|أرخص|ارخص)\s+من\s+(.+)$",
        r"أيهما\s+أفضل[،,:]?\s*(.+?)\s+أم\s+(.+)$",
        r"(?:شنو|شنو هو|ياهو)\s+الأفضل[،,:]?\s*(.+?)\s+(?:ولا|أم)\s+(.+)$",
        r"قارن\s+بين\s+(.+?)\s+و\s*(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None


def comparison_criterion(question):
    cleaned = question.strip(" ؟?")
    match = re.search(
        r"(?:من حيث|من ناحية|في|بالنسبة لـ?|لـ)\s+"
        r"([\u0600-\u06FF\w\s]+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    normalized = question.lower()
    inferred = (
        (("أسهل", "اسهل"), "سهولة الاستخدام أو التعلّم"),
        (("أسرع", "اسرع"), "السرعة"),
        (("أقوى", "اقوى"), "القوة أو الأداء"),
        (("أرخص", "ارخص"), "السعر"),
    )
    return next(
        (
            criterion
            for words, criterion in inferred
            if any(word in normalized for word in words)
        ),
        None,
    )


def difference_request(question):
    cleaned = question.strip(" ؟?")
    match = re.match(
        r"(?:ما\s+)?(?:هو\s+)?الفرق\s+بين\s+(.+?)\s+و(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def normalized_term(text):
    return re.sub(r"^ال", "", text.strip().lower())


def fetch_concept_candidates(term, lang="ar"):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": f'intitle:"{term}"',
        "gsrnamespace": "0",
        "gsrlimit": "6",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "4",
        "redirects": "1",
    }
    try:
        payload = fetch_json(
            f"https://{lang}.wikipedia.org/w/api.php?{urlencode(params)}"
        )
        return payload.get("query", {}).get("pages", [])
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return []


def fetch_exact_concept_pages(term, lang="ar"):
    bare_term = re.sub(r"^ال", "", term.strip())
    titles = [term.strip()]
    if bare_term and bare_term not in titles:
        titles.append(bare_term)
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "titles": "|".join(titles),
        "prop": "extracts|pageprops",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "4",
        "redirects": "1",
    }
    try:
        payload = fetch_json(
            f"https://{lang}.wikipedia.org/w/api.php?{urlencode(params)}",
            attempts=3,
            timeout=10,
        )
        return payload.get("query", {}).get("pages", [])
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return []


def fetch_rest_concept(term, lang="ar"):
    bare_term = re.sub(r"^ال", "", term.strip())
    titles = [term.strip(), bare_term]
    for title in dict.fromkeys(item for item in titles if item):
        try:
            page = fetch_json(
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
                f"{quote(title.replace(' ', '_'))}",
                attempts=2,
                timeout=10,
            )
            extract = page.get("extract", "").strip()
            if (
                page.get("type") != "disambiguation"
                and len(extract) >= 25
            ):
                return {
                    "title": page.get("title", title),
                    "extract": extract,
                    "pageprops": {},
                }
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            continue
    return None


def concept_page_score(page, term):
    title = page.get("title", "").strip()
    extract = " ".join(page.get("extract", "").split())
    if not title or len(extract) < 25:
        return -100
    if "disambiguation" in page.get("pageprops", {}):
        return -100

    normalized_title = normalized_term(
        re.sub(r"\s*\([^)]*\)\s*$", "", title)
    )
    normalized_query = normalized_term(term)
    score = 0
    if normalized_title == normalized_query:
        score += 30
    elif normalized_query in normalized_title.split():
        score += 10
    elif normalized_query in normalized_title:
        score += 5

    query_words = normalize_words(term)
    title_words = normalize_words(title)
    extract_words = normalize_words(extract[:500])
    score += len(query_words & title_words) * 8
    score += len(query_words & extract_words) * 2

    person_signals = (
        r"\bولد\b",
        r"\bتوفي\b",
        r"\(\d{4}\s*[-–]",
        r"\bسياسي\b",
        r"\bلاعب\b",
        r"\bشاعر\b",
        r"\bممثل\b",
        r"\bمواليد\b",
    )
    if any(re.search(pattern, extract[:260]) for pattern in person_signals):
        score -= 35
    unrelated_entity_signals = (
        r"\bمدينة\b",
        r"\bبلدة\b",
        r"\bقرية\b",
        r"\bمحافظة\b",
        r"\bمديرية\b",
        r"\bفيلم\b",
        r"\bمسلسل\b",
        r"\bأغنية\b",
        r"\bجريدة\b",
        r"\bصحيفة\b",
        r"\bمجلة\b",
        r"\bرواية\b",
        r"\bألبوم\b",
    )
    if any(
        re.search(pattern, extract[:220])
        for pattern in unrelated_entity_signals
    ):
        score -= 30
    return score


def wikidata_concept(term):
    params = {
        "action": "wbsearchentities",
        "search": term,
        "language": "ar",
        "uselang": "ar",
        "type": "item",
        "limit": "8",
        "format": "json",
    }
    excluded = (
        "مدينة",
        "قرية",
        "بلدة",
        "فيلم",
        "مسلسل",
        "أغنية",
        "جريدة",
        "صحيفة",
        "مجلة",
        "رواية",
        "كتاب",
        "ألبوم",
        "شركة",
        "منظمة",
        "تطبيق",
        "برنامج",
        "برمجية",
        "موقع إلكتروني",
        "شخص",
        "لاعب",
        "سياسي",
    )
    try:
        results = fetch_json(
            f"https://www.wikidata.org/w/api.php?{urlencode(params)}",
            attempts=2,
        ).get("search", [])
        ranked = []
        for item in results:
            label = item.get("label", "")
            description = item.get("description", "")
            score = 20 if normalized_term(label) == normalized_term(term) else 0
            if any(word in description for word in excluded):
                score -= 25
            if description:
                score += 5
            ranked.append((score, item))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 5:
            return None
        selected = ranked[0][1]
        entity_id = selected["id"]
        entity_params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "sitelinks",
            "sitefilter": "arwiki",
            "format": "json",
        }
        entity = fetch_json(
            f"https://www.wikidata.org/w/api.php?{urlencode(entity_params)}"
        )["entities"][entity_id]
        return {
            "id": entity_id,
            "label": selected.get("label", ""),
            "title": entity.get("sitelinks", {}).get("arwiki", {}).get("title"),
            "description": selected.get("description", "").strip(),
        }
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def concept_summary(extract, term):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!؟])\s+", extract)
        if len(sentence.strip()) >= 20
    ]
    normalized = normalized_term(term)
    ranked = []
    for index, sentence in enumerate(sentences[:6]):
        normalized_sentence = normalized_term(sentence)
        score = -index
        if normalized_sentence.startswith(normalized):
            score += 15
        if re.search(
            rf"\b{re.escape(normalized)}\w*\b.{0,35}\b(?:هو|هي|يعني|عبارة عن)\b",
            normalized_sentence,
        ):
            score += 12
        if normalized in normalized_sentence[:140]:
            score += 5
        if re.search(r"\b(?:كان|كانت|عام|سنة|يظن|اعتقد|مثال)\b", sentence):
            score -= 10
        ranked.append((score, sentence))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked[0][0] >= 4:
            return ranked[0][1]
    return focused_summary(extract, f"ما هو {term}")


def resolve_concept(term):
    cache_key = term.strip().lower()
    if cache_key in CONCEPT_CACHE:
        return CONCEPT_CACHE[cache_key]
    trusted = TRUSTED_CONCEPTS.get(normalized_term(term))
    if trusted:
        result = {
            "term": term,
            "title": trusted["title"],
            "summary": trusted["summary"],
            "source": trusted["source"],
        }
        CONCEPT_CACHE[cache_key] = result
        return result
    wikidata_match = wikidata_concept(term)
    exact_wikidata_label = (
        wikidata_match
        and normalized_term(wikidata_match.get("label", ""))
        == normalized_term(term)
    )
    wikidata_title = wikidata_match.get("title", "") if wikidata_match else ""
    exact_wikidata_title = (
        wikidata_title
        and normalized_term(
            re.sub(r"\s*\([^)]*\)\s*$", "", wikidata_title)
        )
        == normalized_term(term)
        and "(" not in wikidata_title
    )
    wikidata_description = (
        wikidata_match.get("description", "") if wikidata_match else ""
    )
    if (
        exact_wikidata_label
        and exact_wikidata_title
        and wikidata_description
        and re.search(r"[\u0600-\u06FF]", wikidata_description)
    ):
        result = {
            "term": term,
            "title": wikidata_match.get("title") or term,
            "summary": wikidata_description.rstrip(".") + ".",
            "source": f"https://www.wikidata.org/wiki/{wikidata_match['id']}",
        }
        CONCEPT_CACHE[cache_key] = result
        return result
    exact_pages = fetch_exact_concept_pages(term, "ar")
    exact_ranked = sorted(
        exact_pages,
        key=lambda page: concept_page_score(page, term),
        reverse=True,
    )
    if exact_ranked and concept_page_score(exact_ranked[0], term) >= 8:
        candidates = exact_ranked
    else:
        rest_page = fetch_rest_concept(term, "ar")
        candidates = [rest_page] if rest_page else fetch_concept_candidates(term, "ar")
    if not candidates or max(
        (concept_page_score(page, term) for page in candidates),
        default=-100,
    ) < 8:
        concept_title = wikidata_match.get("title") if wikidata_match else None
        candidates = (
            fetch_exact_concept_pages(concept_title, "ar")
            if concept_title
            else []
        )
    if not candidates:
        if wikidata_match and wikidata_match.get("description"):
            return {
                "term": term,
                "title": wikidata_match.get("title") or term,
                "summary": wikidata_match["description"].rstrip(".") + ".",
                "source": (
                    f"https://www.wikidata.org/wiki/{wikidata_match['id']}"
                ),
            }
        return None
    ranked = sorted(
        candidates,
        key=lambda page: concept_page_score(page, term),
        reverse=True,
    )
    best = ranked[0]
    if concept_page_score(best, term) < 8:
        return None
    extract = " ".join(best.get("extract", "").split())
    summary = concept_summary(extract, term)
    description = wikidata_match.get("description") if wikidata_match else ""
    if description and exact_wikidata_label and (
        re.search(
            r"\b(?:كان|كانت|عام|سنة|يظن|فيلم|مدينة|قرية|جريدة|صحيفة|مجلة)\b",
            summary,
        )
    ) and re.search(r"[\u0600-\u06FF]", description):
        summary = description.rstrip(".") + "."
    if not summary:
        return None
    result = {
        "term": term,
        "title": best.get("title", term),
        "summary": summary,
        "source": (
            "https://ar.wikipedia.org/wiki/"
            + quote(best.get("title", term).replace(" ", "_"))
        ),
    }
    CONCEPT_CACHE[cache_key] = result
    return result


DIFFERENCE_DIMENSIONS = (
    ("الطبيعة والبنية", r"خلية|خلوية|جرم|تضاريس|جزء|كائن|عامل|مادة|نظام"),
    ("الوظيفة أو الدور", r"وظيف|يستخدم|استعمال|يعالج|يمارس|لقب|درجة|مهنة|يصف"),
    ("الزمن", r"قصير|طويل|يومي|سنوي|لحظي|مؤقت|دائم|عقود|مدة"),
    ("الحجم والشكل", r"ارتفاع|منحدر|قمة|حجم|أكبر|أصغر|مرتفع|منخفض|ضخم"),
    ("الاستقلال والتكاثر", r"يتكاثر|التكاثر|داخل خلايا|مستقل|ذاتي"),
)
CONTRAST_RULES = (
    (
        "المدة الزمنية",
        r"قصير|يومي|لحظي|حالي|حالة الغلاف الجوي",
        r"طويل|عقود|متوسط حالات|معتاد|النظام المناخي",
        "الأول يصف حالة قصيرة أو آنية، بينما الثاني يصف نمطًا طويل المدى.",
    ),
    (
        "الحجم والارتفاع",
        r"دون|أقل|مصغر|صغير",
        r"شديد الانحدار|ضخم|أعلى|مرتفع",
        "الأول أقل حجمًا أو ارتفاعًا، بينما الثاني أكبر بروزًا أو أشد انحدارًا.",
    ),
    (
        "البنية والتكاثر",
        r"لا يمكنه التكاثر إلا داخل|غير خلوي|عامل ممرض",
        r"وحيدة الخلية|كائنات حية دقيقة|خلية",
        "الأول ليس كائنًا خلويًا مستقلًا ويعتمد على خلية مضيفة للتكاثر، "
        "بينما الثاني كائن خلوي حي يمكنه التكاثر ذاتيًا في الظروف المناسبة.",
    ),
    (
        "الصفة المهنية",
        r"لقب|درجة الدكتوراه|درجة علمية",
        r"درس علم الطب|يمارس الطب|مهنة",
        "أحد المصطلحين قد يدل على لقب أو درجة أكاديمية، بينما الآخر يدل على "
        "مهنة ممارسة الطب.",
    ),
)


def definition_predicate(term, summary):
    text = summary.strip().rstrip(".")
    bare_term = normalized_term(term)
    text = re.sub(
        rf"^(?:ال)?{re.escape(bare_term)}(?:ات|ة)?"
        r"(?:\s+أو\s+[^،]+)?\s*[,،:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:هو|هي|يعني|عبارة عن)\s+", "", text)
    if len(text) > 190:
        text = re.split(r"[؛;]|\s،\s", text, maxsplit=1)[0].strip()
    return text or summary.strip().rstrip(".")


def matching_clause(summary, pattern):
    clauses = [
        clause.strip(" ،؛.")
        for clause in re.split(r"[؛;.]|\s،\s", summary)
        if clause.strip(" ،؛.")
    ]
    return next(
        (clause for clause in clauses if re.search(pattern, clause)),
        None,
    )


def shorten_at_word(text, limit=135):
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].rstrip(" ،؛")
    return shortened + "..."


def analyze_difference(first, second, first_summary, second_summary):
    lines = []
    combined_first = first_summary.lower()
    combined_second = second_summary.lower()
    for dimension, left_pattern, right_pattern, conclusion in CONTRAST_RULES:
        normal_order = re.search(left_pattern, combined_first) and re.search(
            right_pattern, combined_second
        )
        reverse_order = re.search(right_pattern, combined_first) and re.search(
            left_pattern, combined_second
        )
        if normal_order or reverse_order:
            if reverse_order:
                conclusion = conclusion.replace("الأول", "__SECOND__")
                conclusion = conclusion.replace("الثاني", "الأول")
                conclusion = conclusion.replace("__SECOND__", "الثاني")
            lines.append(f"- من حيث {dimension}: {conclusion}")

    if not lines:
        for dimension, pattern in DIFFERENCE_DIMENSIONS:
            first_clause = matching_clause(first_summary, pattern)
            second_clause = matching_clause(second_summary, pattern)
            if first_clause and second_clause:
                first_clause = shorten_at_word(
                    definition_predicate(first, first_clause)
                )
                second_clause = shorten_at_word(
                    definition_predicate(second, second_clause)
                )
                lines.append(
                    f"- من حيث {dimension}: {first} {first_clause}، أما "
                    f"{second} فـ{second_clause}."
                )
                break
    if not lines:
        first_predicate = definition_predicate(first, first_summary)[:150]
        second_predicate = definition_predicate(second, second_summary)[:150]
        lines.append(
            f"- الفرق المباشر: {first} {first_predicate}، بينما "
            f"{second} {second_predicate}."
        )
    return "التحليل:\n" + "\n".join(lines)


def difference_answer(question):
    subjects = difference_request(question)
    if not subjects:
        return None
    first, second = subjects
    first_concept = comparison_concept(first, question)
    second_concept = comparison_concept(second, question)
    if not first_concept or not second_concept:
        time.sleep(0.5)
        first_concept = first_concept or comparison_concept(first, question)
        second_concept = second_concept or comparison_concept(second, question)
    if not first_concept or not second_concept:
        return None
    analysis = analyze_difference(
        first,
        second,
        first_concept["summary"],
        second_concept["summary"],
    )
    return (
        f"الفرق بين {first} و{second}:\n"
        f"- {first}: {first_concept['summary']}\n"
        f"- {second}: {second_concept['summary']}\n"
        f"{analysis}"
        f"\n\nالمصدر: {first_concept['source']}"
        f"\nالمصدر: {second_concept['source']}"
        "\nمرجع البيانات: مقارنة تعريفية من مصدرين"
    )


def comparison_answer(question):
    subjects = comparison_request(question)
    if not subjects:
        return None
    first, second = subjects
    first_concept = comparison_concept(first, question)
    second_concept = comparison_concept(second, question)
    if not first_concept or not second_concept:
        return None

    first_summary = first_concept["summary"]
    second_summary = second_concept["summary"]
    criterion = comparison_criterion(question)
    pair = {first.lower(), second.lower()}
    if pair == {"بايثون", "جافا"}:
        conclusion = (
            "الخلاصة: بايثون أفضل عادةً لسهولة التعلم، والأتمتة، وتحليل البيانات "
            "والذكاء الاصطناعي. أما جافا فتكون أنسب غالبًا للأنظمة المؤسسية "
            "الكبيرة والتطبيقات التي تحتاج بنية صارمة وأداءً ثابتًا. لذلك للمبتدئ "
            "أو لمشروعات البيانات أرجّح بايثون، وللأنظمة المؤسسية أرجّح جافا."
        )
    elif criterion:
        conclusion = (
            f"الخلاصة حسب معيار {criterion}: المعلومات التعريفية بتوضح طبيعة "
            "كل خيار، لكن الحكم النهائي محتاج قياس مباشر للمعيار دا. اختَر "
            "الخيار البحقق احتياجك في المعيار، وما تعتمد على الأفضلية العامة."
        )
    else:
        conclusion = (
            "الخلاصة: ما في خيار أفضل بصورة مطلقة؛ الأفضل بيتحدد حسب استخدامك. "
            f"اختَر {first} لو خصائصو المذكورة أقرب لهدفك، واختَر {second} "
            "لو طبيعتو أنسب للمهمة العايز تعملها."
        )
    analysis = analyze_difference(
        first,
        second,
        first_summary,
        second_summary,
    ).replace("التحليل:", "نقاط المقارنة:")
    return (
        f"{conclusion}\n"
        f"المقارنة بين {first} و{second}:\n"
        f"- {first}: {first_summary}\n"
        f"- {second}: {second_summary}\n"
        f"{analysis}"
        f"\n\nالمصدر: {first_concept['source']}"
        f"\nالمصدر: {second_concept['source']}"
        "\nمرجع البيانات: مقارنة تحليلية من مصدرين"
    )


def wikidata_intent(question):
    normalized = question.strip().lower()
    if re.match(r"^من\s+هو\s+رئيس\s+.+\s+الحالي", normalized):
        return "P35", "الرئيس الحالي هو {value}."
    intents = (
        (("عاصمة",), "P36", "العاصمة هي {value}."),
        (("متى ولد", "تاريخ ميلاد", "ميلاده"), "P569", "وُلد في {value}."),
        (("متى توفي", "تاريخ وفاة", "وفاته"), "P570", "توفي في {value}."),
        (("عدد السكان", "كم عدد سكان", "سكان"), "P1082", "يبلغ عدد السكان {value}."),
        (("مساحة",), "P2046", "تبلغ المساحة {value} كيلومترًا مربعًا."),
        (("العملة", "عملة"), "P38", "العملة هي {value}."),
        (("اللغة الرسمية",), "P37", "اللغة الرسمية هي {value}."),
        (("مؤسس", "أسس"), "P112", "أسسها {value}."),
    )
    for keywords, property_id, template in intents:
        if any(keyword in normalized for keyword in keywords):
            return property_id, template
    return None


def wikidata_subject(question):
    normalized = re.sub(r"[؟?]", "", question.strip().lower())
    removable_phrases = (
        "ما هي عاصمة",
        "ما عاصمة",
        "عاصمة",
        "متى ولد",
        "متى ولدت",
        "ما تاريخ ميلاد",
        "تاريخ ميلاد",
        "متى توفي",
        "متى توفيت",
        "ما تاريخ وفاة",
        "تاريخ وفاة",
        "كم عدد سكان",
        "ما عدد سكان",
        "عدد سكان",
        "ما مساحة",
        "كم مساحة",
        "مساحة",
        "ما عملة",
        "ما هي عملة",
        "عملة",
        "ما اللغة الرسمية في",
        "ما هي اللغة الرسمية في",
        "اللغة الرسمية في",
        "من هو مؤسس شركة",
        "من مؤسس شركة",
        "من هو مؤسس",
        "من مؤسس",
        "مؤسس شركة",
        "مؤسس",
        "من هو رئيس",
        "من رئيس",
        "الرئيس الحالي ل",
        "الرئيس الحالي",
        "رئيس",
        "الحالي",
    )
    for phrase in removable_phrases:
        normalized = normalized.replace(phrase, " ")
    return " ".join(normalized.split())


def wikidata_value(claim):
    value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
    if isinstance(value, dict) and "id" in value:
        return ("entity", value["id"])
    if isinstance(value, dict) and "time" in value:
        date_match = re.match(r"[+-](\d{4})-(\d{2})-(\d{2})", value["time"])
        if date_match:
            year, month, day = map(int, date_match.groups())
            return (
                "text",
                f"{arabic_number(day)} {GREGORIAN_MONTHS[month - 1]} "
                f"{arabic_number(year)}",
            )
    if isinstance(value, dict) and "amount" in value:
        amount = float(value["amount"])
        return ("text", format_number(amount))
    if isinstance(value, (str, int, float)):
        return ("text", format_number(value) if isinstance(value, (int, float)) else value)
    return None


def answer_from_wikidata(question):
    intent = wikidata_intent(question)
    subject = wikidata_subject(question)
    if not intent or not subject:
        return None

    property_id, template = intent
    if property_id == "P36" and subject.strip() in {"الكونغو", "كونغو"}:
        return (
            "هل تقصد جمهورية الكونغو وعاصمتها برازافيل، أم جمهورية "
            "الكونغو الديمقراطية وعاصمتها كينشاسا؟"
        )
    search_params = urlencode(
        {
            "action": "wbsearchentities",
            "search": subject,
            "language": "ar",
            "uselang": "ar",
            "type": "item",
            "limit": "1",
            "format": "json",
        }
    )
    try:
        search_data = fetch_json(
            f"https://www.wikidata.org/w/api.php?{search_params}"
        )
        results = search_data.get("search", [])
        if not results:
            return None
        entity_id = results[0]["id"]
        entity_params = urlencode(
            {
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "claims|sitelinks",
                "languages": "ar",
                "sitefilter": "arwiki",
                "format": "json",
            }
        )
        entity_data = fetch_json(
            f"https://www.wikidata.org/w/api.php?{entity_params}"
        )
        entity = entity_data["entities"][entity_id]
        claims = entity.get("claims", {}).get(property_id, [])
        preferred = [
            claim for claim in claims if claim.get("rank") == "preferred"
        ] or [claim for claim in claims if claim.get("rank") != "deprecated"]
        if not preferred:
            return None
        parsed = wikidata_value(preferred[0])
        if not parsed:
            return None
        value_type, value = parsed
        if value_type == "entity":
            label_params = urlencode(
                {
                    "action": "wbgetentities",
                    "ids": value,
                    "props": "labels",
                    "languages": "ar|en",
                    "format": "json",
                }
            )
            label_data = fetch_json(
                f"https://www.wikidata.org/w/api.php?{label_params}"
            )
            labels = label_data["entities"][value].get("labels", {})
            value = labels.get("ar", labels.get("en", {})).get("value")
            if not value:
                return None
        title = entity.get("sitelinks", {}).get("arwiki", {}).get("title")
        source = (
            f"https://ar.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            if title
            else f"https://www.wikidata.org/wiki/{entity_id}"
        )
        return f"{template.format(value=value)}\n\nالمصدر: {source}"
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def rest_countries_candidate(question):
    intent = wikidata_intent(question)
    subject = wikidata_subject(question)
    if not intent or not subject:
        return None
    property_id, _ = intent
    if property_id not in {"P36", "P1082", "P2046", "P38", "P37"}:
        return None

    try:
        encoded_subject = quote(subject)
        endpoints = (
            f"https://restcountries.com/v3.1/translation/{encoded_subject}",
            f"https://restcountries.com/v3.1/name/{encoded_subject}?fullText=true",
        )
        countries = None
        for endpoint in endpoints:
            try:
                countries = fetch_json(endpoint)
                if countries:
                    break
            except (HTTPError, URLError, TimeoutError, ValueError, OSError):
                continue
        if not countries:
            return None
        country = countries[0]
        source = "https://restcountries.com/"

        if property_id == "P36":
            value = (country.get("capital") or [None])[0]
            answer = f"العاصمة هي {value}." if value else None
        elif property_id == "P2046":
            value = country.get("area")
            answer = (
                f"تبلغ المساحة {format_number(value)} كيلومترًا مربعًا."
                if value is not None
                else None
            )
        elif property_id == "P38":
            currencies = country.get("currencies", {})
            names = [
                item.get("name")
                for item in currencies.values()
                if item.get("name")
            ]
            answer = f"العملة هي {names[0]}." if names else None
        elif property_id == "P37":
            languages = list(country.get("languages", {}).values())
            answer = f"اللغة الرسمية هي {languages[0]}." if languages else None
        else:
            value = country.get("population")
            answer = (
                f"يبلغ عدد السكان نحو {format_number(value)} نسمة."
                if value is not None
                else None
            )
        if not answer:
            return None
        return {
            "answer": answer,
            "source": source,
            "source_name": "REST Countries",
            "score": 88,
            "country_code": country.get("cca3"),
        }
    except (KeyError, TypeError, IndexError):
        return None


def wikidata_country_code(subject):
    params = urlencode(
        {
            "action": "wbsearchentities",
            "search": subject,
            "language": "ar",
            "uselang": "ar",
            "type": "item",
            "limit": "1",
            "format": "json",
        }
    )
    try:
        search_data = fetch_json(f"https://www.wikidata.org/w/api.php?{params}")
        results = search_data.get("search", [])
        if not results:
            return None
        entity_id = results[0]["id"]
        entity_params = urlencode(
            {
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "claims",
                "format": "json",
            }
        )
        entity_data = fetch_json(
            f"https://www.wikidata.org/w/api.php?{entity_params}"
        )
        claims = entity_data["entities"][entity_id].get("claims", {}).get("P298", [])
        if not claims:
            return None
        return claims[0]["mainsnak"]["datavalue"]["value"]
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def rest_countries_candidate_by_code(question, country_code):
    intent = wikidata_intent(question)
    if not intent or not country_code:
        return None
    property_id, _ = intent
    if property_id not in {"P36", "P1082", "P2046", "P38", "P37"}:
        return None
    try:
        countries = fetch_json(
            f"https://restcountries.com/v3.1/alpha/{quote(country_code)}"
        )
        if not countries:
            return None
        country = countries[0]
        source = "https://restcountries.com/"
        if property_id == "P36":
            value = (country.get("capital") or [None])[0]
            answer = f"العاصمة هي {value}." if value else None
        elif property_id == "P2046":
            value = country.get("area")
            answer = (
                f"تبلغ المساحة {format_number(value)} كيلومترًا مربعًا."
                if value is not None
                else None
            )
        elif property_id == "P38":
            values = [
                item.get("name")
                for item in country.get("currencies", {}).values()
                if item.get("name")
            ]
            answer = f"العملة هي {values[0]}." if values else None
        elif property_id == "P37":
            values = list(country.get("languages", {}).values())
            answer = f"اللغة الرسمية هي {values[0]}." if values else None
        else:
            value = country.get("population")
            answer = (
                f"يبلغ عدد السكان نحو {format_number(value)} نسمة."
                if value is not None
                else None
            )
        if not answer:
            return None
        return {
            "answer": answer,
            "source": source,
            "source_name": "REST Countries",
            "score": 88,
            "country_code": country.get("cca3") or country_code,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def world_bank_population_candidate(question, country_code):
    intent = wikidata_intent(question)
    if not intent or intent[0] != "P1082" or not country_code:
        return None
    params = urlencode({"format": "json", "per_page": "30"})
    url = (
        f"https://api.worldbank.org/v2/country/{quote(country_code)}"
        f"/indicator/SP.POP.TOTL?{params}"
    )
    try:
        payload = fetch_json(url)
        records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        record = next(
            (item for item in records if item.get("value") is not None),
            None,
        )
        if not record:
            return None
        return {
            "answer": (
                f"بلغ عدد السكان {format_number(record['value'])} نسمة "
                f"في عام {arabic_number(record['date'])}."
            ),
            "source": "https://data.worldbank.org/indicator/SP.POP.TOTL",
            "source_name": "البنك الدولي",
            "score": 100,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def world_bank_gdp_candidate(question):
    normalized = question.lower()
    if not any(
        phrase in normalized
        for phrase in ("الناتج المحلي", "الناتج الإجمالي", "الناتج الاجمالي")
    ):
        return None
    subject = normalized
    for phrase in (
        "كم يبلغ الناتج المحلي الإجمالي",
        "كم يبلغ الناتج المحلي",
        "ما الناتج المحلي الإجمالي",
        "ما الناتج المحلي",
        "الناتج المحلي الإجمالي",
        "الناتج المحلي",
        "لـ",
    ):
        subject = subject.replace(phrase, " ")
    subject = re.sub(r"^(?:في|لدولة)\s+", "", subject.strip())
    subject = re.sub(r"^لل(?=[\u0600-\u06FF])", "ال", subject)
    subject = re.sub(r"^ل(?=ال[\u0600-\u06FF])", "", subject)
    subject = " ".join(subject.strip(" ؟?").split())
    country_code = wikidata_country_code(subject)
    if not country_code:
        return None
    params = urlencode({"format": "json", "per_page": "20"})
    url = (
        f"https://api.worldbank.org/v2/country/{quote(country_code)}"
        f"/indicator/NY.GDP.MKTP.CD?{params}"
    )
    try:
        payload = fetch_json(url)
        records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        record = next((item for item in records if item.get("value") is not None), None)
        if not record:
            return None
        value = float(record["value"])
        if value >= 1_000_000_000:
            amount = f"{format_number(round(value / 1_000_000_000, 2))} مليار دولار"
        else:
            amount = f"{format_number(round(value / 1_000_000, 2))} مليون دولار"
        return (
            f"بلغ الناتج المحلي الإجمالي {amount} في عام "
            f"{arabic_number(record['date'])}."
            "\n\nالمصدر: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD"
            "\nمرجع البيانات: البنك الدولي"
        )
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def multi_source_structured_answer(question):
    gdp_answer = world_bank_gdp_candidate(question)
    if gdp_answer:
        return gdp_answer

    candidates = []
    intent = wikidata_intent(question)
    country_code = None
    if intent and intent[0] == "P1082":
        country_code = wikidata_country_code(wikidata_subject(question))
        world_bank_candidate = world_bank_population_candidate(
            question, country_code
        )
        if world_bank_candidate:
            return (
                f"{world_bank_candidate['answer']}\n\n"
                f"المصدر: {world_bank_candidate['source']}\n"
                f"مرجع البيانات: {world_bank_candidate['source_name']}"
            )

    rest_candidate = rest_countries_candidate(question)
    if rest_candidate:
        if not intent or intent[0] != "P1082":
            return (
                f"{rest_candidate['answer']}\n\n"
                f"المصدر: {rest_candidate['source']}\n"
                f"مرجع البيانات: {rest_candidate['source_name']}"
            )
        candidates.append(rest_candidate)
    elif intent:
        country_code = country_code or wikidata_country_code(
            wikidata_subject(question)
        )
        rest_candidate = rest_countries_candidate_by_code(
            question, country_code
        )
        if rest_candidate:
            candidates.append(rest_candidate)

    wikidata_answer = answer_from_wikidata(question)
    if wikidata_answer and "\n\nالمصدر: " in wikidata_answer:
        answer, source = wikidata_answer.split("\n\nالمصدر: ", 1)
        candidates.append(
            {
                "answer": answer,
                "source": source,
                "source_name": "ويكي بيانات",
                "score": 82,
            }
        )
    elif wikidata_answer:
        return wikidata_answer

    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: candidate["score"])
    return (
        f"{best['answer']}\n\nالمصدر: {best['source']}"
        f"\nمرجع البيانات: {best['source_name']}"
    )


def sentence_relevance(sentence, question):
    question_words = normalize_words(question)
    sentence_words = normalize_words(sentence)
    overlap = len(question_words & sentence_words)
    stem_overlap = sum(
        1
        for question_word in question_words
        if len(question_word) >= 4
        and any(
            sentence_word.startswith(question_word[:4])
            or question_word.startswith(sentence_word[:4])
            for sentence_word in sentence_words
            if len(sentence_word) >= 4
        )
    )
    normalized_question = question.strip().lower()
    score = overlap * 4 + stem_overlap * 2

    if normalized_question.startswith(("من ", "من هو", "من هي")):
        if re.search(r"\b(هو|هي|كان|كانت|سياسي|عالم|كاتب|رئيس)\b", sentence):
            score += 3
    if normalized_question.startswith(("متى", "في أي عام", "في اي عام")):
        if re.search(r"\d{3,4}|عام|سنة|تاريخ", sentence):
            score += 5
    if normalized_question.startswith(("أين", "اين", "في أي مكان", "في اي مكان")):
        if re.search(r"\b(تقع|يقع|عاصمة|منطقة|شمال|جنوب|شرق|غرب|وسط)\b", sentence):
            score += 7
    if normalized_question.startswith(("ما ", "ماذا", "ما هي", "ما هو")):
        if re.search(r"\b(هو|هي|عبارة عن|يعني|تعرف)\b", sentence):
            score += 3
    if any(word in normalized_question for word in ("استخدام", "تستخدم", "فائدة")):
        if re.search(r"\b(استخدام|تستخدم|تقنيات|تطبيقات|تستعمل)\w*", sentence):
            score += 8
    if any(word in normalized_question for word in ("علاج", "دواء", "خفض")):
        if re.search(
            r"\b(علاج|يعالج|دواء|أدوية|خفض|تقليل|نمط الحياة|الرياضة|الملح)\w*",
            sentence,
        ):
            score += 12

    return score


def page_relevance_score(page, question):
    title = page.get("title", "")
    extract = page.get("extract", "")
    question_words = normalize_words(question)
    title_words = normalize_words(title)
    extract_words = normalize_words(extract[:1200])
    exact_title_overlap = len(question_words & title_words)
    content_overlap = len(question_words & extract_words)
    stem_overlap = sum(
        1
        for question_word in question_words
        if len(question_word) >= 4
        and any(
            word.startswith(question_word[:4])
            or question_word.startswith(word[:4])
            for word in title_words | extract_words
            if len(word) >= 4
        )
    )
    return exact_title_overlap * 6 + content_overlap * 2 + stem_overlap


def focused_summary(text, question):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!؟])\s+", text)
        if len(sentence.strip()) >= 25
    ]
    if not sentences:
        return ""

    normalized_question = question.strip().lower()
    if "عاصمة" in normalized_question:
        capital_patterns = (
            r"(?:عاصمتها|عاصمة\s+\S+(?:\s+\S+){0,3}\s+هي)\s+"
            r"([\u0600-\u06FF][\u0600-\u06FF\s-]{1,40})",
            r"العاصمة\s+(?:هي\s+)?"
            r"([\u0600-\u06FF][\u0600-\u06FF\s-]{1,40})",
        )
        for pattern in capital_patterns:
            capital_match = re.search(pattern, text)
            if capital_match:
                capital = re.split(
                    r"[،.;؛]|\s(?:وتقع|وهي|التي)\s",
                    capital_match.group(1),
                )[0].strip()
                if capital:
                    return f"العاصمة هي {capital}."
    if normalized_question.startswith("متى") and any(
        word in normalized_question for word in ("ولد", "ولدت", "ميلاد")
    ):
        date_match = re.search(
            r"\((\d{1,2}\s+\S+\s+\d{4})\s*[-–]",
            text,
        )
        if date_match:
            return f"وُلد في {date_match.group(1)}."

    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (sentence_relevance(item[1], question), -item[0]),
        reverse=True,
    )
    answer = ranked[0][1]
    normalized_question = question.strip().lower()
    if normalized_question.startswith(("ما هي", "ما هو", "ما ")):
        definition_end = re.search(r"[،؛;]", answer)
        if definition_end and definition_end.start() >= 35:
            concise_definition = answer[: definition_end.start()].strip()
            if len(concise_definition) >= 35:
                answer = concise_definition
    if len(answer) > 320:
        clauses = [
            clause.strip(" ،؛")
            for clause in re.split(r"[؛;]|\s،\s", answer)
            if clause.strip(" ،؛")
        ]
        relevant_clauses = sorted(
            enumerate(clauses),
            key=lambda item: (sentence_relevance(item[1], question), -item[0]),
            reverse=True,
        )
        answer = relevant_clauses[0][1] if relevant_clauses else answer
    answer = answer.rstrip("،؛: ")
    if answer and answer[-1] not in ".!؟":
        answer += "."
    return answer


def fetch_wikipedia_page(search_term, lang="ar"):
    pages = fetch_wikipedia_pages(search_term, lang=lang, limit=1)
    return pages[0] if pages else None


def fetch_wikipedia_pages(search_term, lang="ar", limit=5):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": "0",
        "gsrlimit": str(limit),
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "8",
        "redirects": "1",
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?{urlencode(params)}"
    try:
        payload = fetch_json(url)
        pages = payload.get("query", {}).get("pages", [])
        return sorted(
            pages,
            key=lambda page: page_relevance_score(page, search_term),
            reverse=True,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return []


def evidence_summary(pages, question, max_points=2):
    ranked_sentences = []
    seen = set()
    for page_index, page in enumerate(pages):
        extract = " ".join(page.get("extract", "").split())
        for sentence_index, sentence in enumerate(
            re.split(r"(?<=[.!؟])\s+", extract)
        ):
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            score = sentence_relevance(sentence, question)
            score += max(0, 4 - page_index * 2)
            score -= sentence_index * 0.15
            ranked_sentences.append((score, sentence, page))

    ranked_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = []
    selected_pages = []
    selected_words = set()
    for score, sentence, page in ranked_sentences:
        words = normalize_words(sentence)
        if selected and len(words & selected_words) >= max(5, len(words) // 2):
            continue
        selected.append(shorten_at_word(sentence, 260))
        selected_pages.append(page)
        selected_words |= words
        if len(selected) >= max_points:
            break
    return selected, selected_pages


def comparison_concept(term, question):
    domain = question_domain(question)
    concept = resolve_concept(term)
    domain_signals = {
        "technology": r"برمج|تقني|حاسوب|برنامج|لغة|نظام|تطبيق",
        "medical": r"طب|صح|مرض|علاج|دواء|جسم",
        "science": r"علم|طبيع|فيزي|كيمي|أحياء|مناخ|طقس",
        "economy": r"اقتصاد|مال|سوق|شركة|تجارة|استثمار",
    }
    required_signal = domain_signals.get(domain)
    if concept and (
        not required_signal
        or re.search(
            required_signal,
            f"{concept.get('title', '')} {concept.get('summary', '')}",
            flags=re.IGNORECASE,
        )
    ):
        return concept

    qualifiers = {
        "technology": ("لغة برمجة", "تقنية", "برنامج"),
        "medical": ("طب", "صحة"),
        "science": ("علم",),
        "economy": ("اقتصاد", "شركة"),
    }.get(domain, ())
    aliases = {
        ("technology", "جافا"): "جافا لغة برمجة",
        ("technology", "بايثون"): "بايثون لغة برمجة",
    }
    preferred = aliases.get((domain, normalized_term(term)))
    search_terms = ([preferred] if preferred else []) + [term] + [
        f"{term} {qualifier}" for qualifier in qualifiers
    ]
    for search_term in dict.fromkeys(search_terms):
        pages = fetch_wikipedia_pages(search_term, "ar", limit=5)
        ranked = sorted(
            pages,
            key=lambda page: page_relevance_score(
                page, f"{term} {question}"
            ),
            reverse=True,
        )
        if not ranked or page_relevance_score(ranked[0], term) < 3:
            continue
        page = ranked[0]
        extract = " ".join(page.get("extract", "").split())
        summary = focused_summary(extract, f"ما هو {term}")
        if not summary:
            continue
        title = page.get("title", term)
        return {
            "term": term,
            "title": title,
            "summary": summary,
            "source": (
                "https://ar.wikipedia.org/wiki/"
                + quote(title.replace(" ", "_"))
            ),
        }
    return None


def dictionary_term(question):
    match = re.search(
        r"(?:معنى|مرادف|ضد|تعريف|ترجمة)\s+(?:كلمة\s+)?[\"«]?"
        r"([\w\u0600-\u06FF-]+)",
        question,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip("\"»؟?.,،؛;:") if match else None


def fetch_wiktionary_answer(question):
    term = dictionary_term(question)
    if not term:
        return None
    lang = "en" if re.fullmatch(r"[A-Za-z-]+", term) else "ar"
    if lang == "en":
        try:
            entries = fetch_json(
                f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(term)}"
            )
            for entry in entries:
                for meaning in entry.get("meanings", []):
                    definitions = meaning.get("definitions", [])
                    if definitions and definitions[0].get("definition"):
                        definition = definitions[0]["definition"].strip()
                        return (
                            f"{term}: {definition}\n\n"
                            f"المصدر: https://dictionaryapi.dev/ "
                            f"(مصدر معجمي أجنبي)"
                        )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ):
            pass
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "exsentences": "3",
        "redirects": "1",
        "titles": term,
    }
    try:
        payload = fetch_json(
            f"https://{lang}.wiktionary.org/w/api.php?{urlencode(params)}"
        )
        pages = payload.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        extract = " ".join(pages[0].get("extract", "").split())
        if len(extract) < 20:
            return None
        summary = focused_summary(extract, question) or extract[:280].rstrip()
        source = (
            f"https://{lang}.wiktionary.org/wiki/"
            f"{quote(term.replace(' ', '_'))}"
        )
        foreign_note = " (مصدر أجنبي)" if lang == "en" else ""
        return f"{summary}\n\nالمصدر: {source}{foreign_note}"
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def foreign_wikipedia_page(search_term):
    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "ar",
        "uselang": "en",
        "type": "item",
        "limit": "1",
        "format": "json",
    }
    try:
        result = fetch_json(
            f"https://www.wikidata.org/w/api.php?{urlencode(params)}"
        ).get("search", [])
        if not result:
            return fetch_wikipedia_page(search_term, "en")
        entity_id = result[0]["id"]
        entity_params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json",
        }
        entity = fetch_json(
            f"https://www.wikidata.org/w/api.php?{urlencode(entity_params)}"
        )["entities"][entity_id]
        title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
        return fetch_wikipedia_page(title, "en") if title else None
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def get_arabic_title_from_en(en_title):
    params = {
        "action": "query",
        "titles": en_title,
        "prop": "langlinks",
        "lllang": "ar",
        "format": "json",
        "formatversion": "2",
    }
    url = f"https://en.wikipedia.org/w/api.php?{urlencode(params)}"
    try:
        data = fetch_json(url)
        pages = data.get("query", {}).get("pages", [])
        if pages and "langlinks" in pages[0]:
            return pages[0]["langlinks"][0]["title"]
    except:
        pass
    return None


def scientific_search_request(question):
    normalized = question.lower()
    return bool(
        re.search(
            r"(?:ال)?(?:بحث|أبحاث|ابحاث|دراسة|دراسات|ورقة|أوراق)"
            r"\s+(?:ال)?(?:علمي|علمية|حديث|حديثة)"
            r"|(?:ال)?دراسات\s+(?:ال)?حديثة"
            r"|(?:ال)?أبحاث\s+(?:ال)?حديثة"
            r"|ما\s+يقوله\s+العلم|الدليل\s+العلمي",
            normalized,
        )
    )


def scientific_search_terms(question):
    cleaned = re.sub(
        r"\b(?:أحدث|احدث|آخر|اخر|البحث|بحث|الأبحاث|أبحاث|الابحاث|ابحاث|"
        r"العلمي|علمي|العلمية|علمية|الدراسة|دراسة|الدراسات|دراسات|"
        r"الورقة|ورقة|الأوراق|أوراق|الحديثة|الحديث|الدليل|ما يقوله العلم)\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[؟?!.,،؛:«»\"'()]+", " ", cleaned)
    raw_words = re.sub(r"[^\w\u0600-\u06FF]+", " ", cleaned).split()
    translated = []
    untranslated = []
    for raw_word in raw_words:
        candidates = (raw_word, raw_word.removeprefix("و"))
        translation = next(
            (
                SCIENTIFIC_TERM_TRANSLATIONS[candidate]
                for candidate in candidates
                if candidate in SCIENTIFIC_TERM_TRANSLATIONS
            ),
            None,
        )
        if translation:
            translated.append(translation)
        elif raw_word not in ARABIC_STOP_WORDS and len(raw_word) > 1:
            untranslated.append(raw_word)
    english_terms = re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", question)
    if translated or english_terms:
        return " ".join(dict.fromkeys(translated + english_terms))
    return " ".join(untranslated) or internet_search_terms(cleaned)


def scientific_topic_label(question):
    cleaned = re.sub(
        r"\b(?:ما|هي|هو|عن|أحدث|احدث|آخر|اخر|البحث|بحث|الأبحاث|"
        r"أبحاث|الابحاث|ابحاث|العلمي|علمي|العلمية|علمية|الدراسة|"
        r"دراسة|الدراسات|دراسات|الحديثة|الحديث)\b",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ؟?،") or "الموضوع المطلوب"


def openalex_abstract(work):
    inverted = work.get("abstract_inverted_index") or {}
    positioned = []
    for word, positions in inverted.items():
        positioned.extend((position, word) for position in positions)
    positioned.sort()
    return " ".join(word for _, word in positioned)


def scientific_work_score(work, query):
    title = work.get("display_name", "")
    abstract = openalex_abstract(work)
    overlap = len(
        normalize_words(query)
        & normalize_words(f"{title} {abstract[:1600]}")
    )
    cited = min(int(work.get("cited_by_count") or 0), 500) / 100
    year = int(work.get("publication_year") or 0)
    recency = max(0, year - 2018) * 0.35
    has_abstract = 4 if abstract else 0
    evidence_boost = 0
    if re.search(r"systematic review|meta-analysis", title, re.IGNORECASE):
        evidence_boost = 8
    elif re.search(r"randomi[sz]ed controlled trial", title, re.IGNORECASE):
        evidence_boost = 5
    return overlap * 5 + cited + recency + has_abstract + evidence_boost


def scientific_query_coverage(work, query):
    query_words = normalize_words(query)
    if not query_words:
        return 0.0
    content_words = normalize_words(
        f"{work.get('display_name', '')} {openalex_abstract(work)[:2000]}"
    )
    return len(query_words & content_words) / len(query_words)


def scientific_finding(work, query):
    abstract = openalex_abstract(work)
    if not abstract:
        return "الملخص الكامل ما متاح في الفهرس، لذلك ما بننسب ليها نتيجة محددة."
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", abstract)
        if len(item.strip()) >= 45
    ]
    conclusion_sentences = [
        sentence
        for sentence in sentences
        if re.search(
            r"\b(?:conclud|result|found|suggest|indicat|associated|"
            r"improv|increase|decrease|effect)\w*",
            sentence,
            flags=re.IGNORECASE,
        )
    ]
    candidates = conclusion_sentences or sentences
    if not candidates:
        return "الملخص المتاح ما كفاية لاستخراج نتيجة واضحة."
    best = max(candidates, key=lambda sentence: sentence_relevance(sentence, query))
    best = shorten_at_word(best, 155)
    lower = best.lower()
    query_words = normalize_words(query)
    if {"exercise", "depression"} <= query_words:
        if "effective treatment" in lower or "significant reduction" in lower:
            return (
                "الملخص بيدعم إن التمارين بتقلل أعراض الاكتئاب، "
                "مع اختلاف الأثر حسب نوع التمرين وشدتو والفئة المدروسة."
            )
    if {"sleep", "memory"} <= query_words:
        if "memory consolidation" in lower or "critical for memory" in lower:
            return "الملخص بيوضح إن النوم مهم لتثبيت الذكريات بعد التعلّم."
        if "sleep deprivation" in lower or "total sleep deprivation" in lower:
            return (
                "الملخص بربط الحرمان من النوم بضعف جوانب من الذاكرة "
                "والأداء المعرفي."
            )
        if "cognitive impairment" in lower and "improve" in lower:
            return (
                "الدراسة لقت تحسناً في الضعف المعرفي المرتبط بحرمان النوم "
                "داخل النموذج البدرسو."
            )
    return f"النص الأوضح في الملخص المتاح: {best}"


def scientific_evidence_type(work):
    text = (
        f"{work.get('display_name', '')} {openalex_abstract(work)[:500]}"
    ).lower()
    if "systematic review" in text and "meta-analysis" in text:
        return "مراجعة منهجية وتحليل تجميعي"
    if "meta-analysis" in text:
        return "تحليل تجميعي"
    if "systematic review" in text:
        return "مراجعة منهجية"
    if re.search(r"randomi[sz]ed controlled trial", text):
        return "تجربة عشوائية محكومة"
    if "cohort" in text:
        return "دراسة أترابية"
    if "cross-sectional" in text:
        return "دراسة مقطعية"
    return "دراسة بحثية"


def scientific_consensus(works, query):
    combined = " ".join(
        f"{work.get('display_name', '')} {openalex_abstract(work)}".lower()
        for work in works
    )
    query_words = normalize_words(query)
    if {"exercise", "depression"} <= query_words:
        if "effective treatment" in combined or "reduction in levels of depression" in combined:
            return (
                "الخلاصة العلمية: الأدلة المعروضة بتميل بوضوح إلى إن النشاط "
                "البدني بساعد في تقليل أعراض الاكتئاب، لكن نوع التمرين وشدتو "
                "وحالة الشخص عوامل مهمة."
            )
    if {"sleep", "memory"} <= query_words:
        if "memory consolidation" in combined or "sleep deprivation" in combined:
            return (
                "الخلاصة العلمية: الاتجاه العام بيوضح إن النوم الجيد بدعم "
                "تثبيت الذاكرة، وإن الحرمان من النوم ممكن يضعف الذاكرة العاملة "
                "وبعض الوظائف المعرفية."
            )
    return (
        "الخلاصة العلمية: الدراسات مرتبطة بالسؤال، لكن قوة النتيجة بتعتمد "
        "على نوع الدراسة وحجم العينة؛ عشان كدا النتيجة اتجاه مدعوم، ما يقين مطلق."
    )


def scientific_search_answer(question):
    if not scientific_search_request(question):
        return None
    query = scientific_search_terms(question)
    if not query:
        return None
    params = urlencode(
        {
            "search": query,
            "per-page": "20",
            "select": (
                "id,doi,display_name,publication_year,publication_date,"
                "cited_by_count,abstract_inverted_index,primary_location"
            ),
        }
    )
    if any(word in question.lower() for word in ("أحدث", "احدث", "الحديثة")):
        params += "&filter=from_publication_date:2021-01-01"
    try:
        payload = fetch_json(
            f"https://api.openalex.org/works?{params}",
            attempts=2,
            timeout=12,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None
    works = payload.get("results", [])
    ranked = sorted(
        works,
        key=lambda work: scientific_work_score(work, query),
        reverse=True,
    )
    selected = [
        work
        for work in ranked
        if scientific_work_score(work, query) >= 9
        and scientific_query_coverage(work, query) >= 0.6
    ][:3]
    if not selected:
        return None

    topic_label = scientific_topic_label(question)
    lines = [
        f"لقيت {len(selected)} دراسات مرتبطة مباشرة بسؤالك عن {topic_label}:"
    ]
    sources = []
    findings_with_abstract = 0
    for index, work in enumerate(selected, start=1):
        title = work.get("display_name") or "دراسة بلا عنوان ظاهر"
        year = work.get("publication_year") or "غير محدد"
        finding = scientific_finding(work, query)
        evidence_type = scientific_evidence_type(work)
        if work.get("abstract_inverted_index"):
            findings_with_abstract += 1
        lines.append(
            f"{index}. {title} ({year}).\n"
            f"   نوع الدليل: {evidence_type}.\n"
            f"   النتيجة المختصرة: {finding}"
        )
        source = work.get("doi") or work.get("id")
        if source:
            sources.append(f"المصدر: {source}")

    if findings_with_abstract >= 2:
        lines.append(scientific_consensus(selected, query))
    else:
        lines.append(
            "الخلاصة العلمية: البيانات الببليوغرافية متاحة، لكن الملخصات "
            "الكاملة قليلة؛ لذلك الأفضل اعتبار القائمة نقطة بداية للمراجعة، "
            "ما حكم نهائي."
        )
    return (
        "\n".join(lines)
        + "\n\n"
        + "\n".join(dict.fromkeys(sources))
        + "\nمرجع البيانات: OpenAlex، فهرس أبحاث علمية مفتوح"
    )


def search_internet(prompt):
    domain = question_domain(prompt)
    scientific_answer = scientific_search_answer(prompt)
    if scientific_answer:
        return scientific_answer
    if domain == "dictionary":
        dictionary_answer = fetch_wiktionary_answer(prompt)
        if dictionary_answer:
            return dictionary_answer

    latest_answer = latest_python_answer(prompt)
    if latest_answer:
        return latest_answer

    normalized = prompt.lower()
    if domain == "medical" and any(
        phrase in normalized
        for phrase in ("ارتفاع ضغط الدم", "ضغط الدم المرتفع", "فرط ضغط الدم")
    ) and any(word in normalized for word in ("علاج", "خفض", "دواء")):
        return (
            "يُعالج ارتفاع ضغط الدم بتقليل الملح، والنشاط البدني، وضبط الوزن، "
            "والامتناع عن التدخين، وقد يصف الطبيب أدوية خافضة للضغط حسب الحالة. "
            "لا تبدأ دواءً أو توقفه دون استشارة طبية."
            "\n\nالمصدر: https://www.who.int/news-room/fact-sheets/detail/hypertension"
            "\nمرجع البيانات: منظمة الصحة العالمية"
        )

    # 1. محاولة الحصول على إجابة مهيكلة أولاً
    structured_answer = multi_source_structured_answer(prompt)
    if structured_answer:
        return structured_answer

    search_term = internet_search_terms(prompt)
    
    # 2. إعطاء الأولوية للعنوان المطابق ثم ترتيب النتائج حسب السؤال
    exact_pages = fetch_exact_concept_pages(search_term, "ar")
    search_pages = fetch_wikipedia_pages(search_term, "ar", limit=5)
    pages_by_title = {}
    for candidate in exact_pages + search_pages:
        title_key = candidate.get("title", "").strip().lower()
        if title_key:
            pages_by_title[title_key] = candidate
    pages = sorted(
        pages_by_title.values(),
        key=lambda candidate: (
            concept_page_score(candidate, search_term),
            page_relevance_score(candidate, prompt),
        ),
        reverse=True,
    )
    source_lang = "ar"
    note = ""

    # 3. إذا لم توجد نتيجة كافية، ابحث في الإنجليزية
    if not pages or page_relevance_score(pages[0], prompt) < 5:
        en_page = foreign_wikipedia_page(search_term)
        if en_page:
            # محاولة العثور على النسخة العربية للمقال الإنجليزي
            ar_title = get_arabic_title_from_en(en_page.get("title", ""))
            if ar_title:
                pages = fetch_wikipedia_pages(ar_title, "ar", limit=3)
            else:
                # استخدام المحتوى الإنجليزي كخيار أخير
                pages = [en_page]
                source_lang = "en"
                note = " (مصدر أجنبي)"

    if not pages:
        return random.choice(UNKNOWN_RESPONSES)

    page = pages[0]
    title = page.get("title", "").strip()
    if page_relevance_score(page, prompt) < 5:
        return (
            "لم أجد مصدرًا يجيب عن السؤال نفسه بدرجة كافية، لذلك لن أعرض "
            "نتيجة بعيدة أو غير مرتبطة به."
        )

    definition_question = prompt.strip().lower().startswith(
        ("ما هو", "ما هي", "ما معنى", "عرّف", "عرف")
    )
    points, evidence_pages = evidence_summary(
        pages[:3],
        prompt,
        max_points=1 if definition_question else 2,
    )
    if source_lang == "en":
        extract = " ".join(page.get("extract", "").split())
        summary = focused_summary(extract, prompt) or shorten_at_word(extract, 320)
    elif points:
        summary = points[0]
        if len(points) > 1:
            summary += f"\nوالنقطة المهمة كمان: {points[1]}"
    else:
        extract = " ".join(page.get("extract", "").split())
        summary = focused_summary(extract, prompt)

    if not summary:
        return random.choice(UNKNOWN_RESPONSES)

    source_url = f"https://{source_lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    source_type = DOMAIN_SOURCE_NAMES.get(domain, "مرجع عام")
    source_lines = [f"المصدر: {source_url}{note}"]
    for evidence_page in evidence_pages[1:2]:
        evidence_title = evidence_page.get("title", "").strip()
        if evidence_title and evidence_title != title:
            source_lines.append(
                f"المصدر: https://{source_lang}.wikipedia.org/wiki/"
                f"{quote(evidence_title.replace(' ', '_'))}{note}"
            )
    return (
        f"{summary}\n\n" + "\n".join(source_lines)
        + f"\nمرجع البيانات: {source_type}، مع ترتيب النتائج حسب صلة السؤال"
    )


def extract_topic(question):
    cleaned = re.sub(r"[؟?]", "", question.strip())
    patterns = (
        r"^(?:من هو|من هي|ما هو|ما هي)\s+(.+)$",
        r"^(?:متى ولد|متى ولدت|متى توفي|متى توفيت)\s+(.+)$",
        r"^(?:أين تقع|اين تقع|أين يقع|اين يقع)\s+(.+)$",
        r"^(?:ما عاصمة|ما هي عاصمة|ما عملة|ما هي عملة|ما مساحة)\s+(.+)$",
        r"^(?:كم عدد سكان)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def resolve_follow_up(prompt, history):
    if not history:
        return prompt

    normalized = prompt.strip().lower()
    reference_markers = (
        "هو",
        "هي",
        "له",
        "لها",
        "عنه",
        "عنها",
        "عملته",
        "عملتها",
        "عاصمته",
        "عاصمتها",
        "مساحته",
        "مساحتها",
        "سكانه",
        "سكانها",
    )
    words = normalize_words(normalized)
    if len(words) >= 3 and not any(
        marker in normalized for marker in reference_markers
    ):
        return prompt

    topic = None
    for item in reversed(history[-6:]):
        if item.get("role") != "user":
            continue
        topic = extract_topic(item.get("content", ""))
        if topic:
            break
    if not topic:
        return prompt

    replacements = (
        (r"\b(?:هو|هي)\b", topic),
        (r"\b(?:له|لها)\b", f"لـ {topic}"),
        (r"\b(?:عنه|عنها)\b", f"عن {topic}"),
        (r"\b(?:عملته|عملتها)\b", f"عملة {topic}"),
        (r"\b(?:عاصمته|عاصمتها)\b", f"عاصمة {topic}"),
        (r"\b(?:مساحته|مساحتها)\b", f"مساحة {topic}"),
        (r"\b(?:سكانه|سكانها)\b", f"سكان {topic}"),
    )
    resolved = prompt
    for pattern, replacement in replacements:
        resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)

    if resolved == prompt and len(words) <= 3:
        resolved = f"{prompt.rstrip('؟?')} {topic}؟"
    return resolved


def assistant_meta_response(prompt):
    normalized = re.sub(r"[؟?!.،]", "", prompt.strip().lower())
    identity_phrases = (
        "من انت",
        "من أنت",
        "عرف نفسك",
        "عرّف نفسك",
        "انت منو",
        "أنت منو",
        "إنت منو",
        "شنو انت",
        "شنو أنت",
    )
    capability_phrases = (
        "ما الذي تستطيع فعله",
        "ماذا تستطيع أن تفعل",
        "ماذا تستطيع فعله",
        "ما هي قدراتك",
        "ما قدراتك",
        "شنو بتقدر تعمل",
        "بتقدر تعمل شنو",
        "تقدر تعمل شنو",
        "بتساعدني كيف",
    )
    asks_identity = any(phrase in normalized for phrase in identity_phrases)
    asks_capabilities = any(
        phrase in normalized for phrase in capability_phrases
    )
    if asks_identity and asks_capabilities:
        return (
            "أنا مساعد لغوي بدون اسم، بتكلم معاك بالدارجية السودانية. "
            "بقدر أجاوب على الأسئلة، وأبحث في النت، وأقارن وأحلل، وأحسب، "
            "وأتعامل مع أكتر من سؤال، وأقرأ الرد بصوت سوداني."
        )
    if asks_identity:
        return (
            "أنا مساعد لغوي بدون اسم، بتكلم معاك بالدارجية السودانية "
            "وبساعدك في الأسئلة والبحث والتحليل."
        )
    if asks_capabilities:
        return (
            "أستطيع الإجابة عن الأسئلة، والبحث في الإنترنت، والمقارنة والتحليل، "
            "وحل العمليات الرياضية، ومعالجة عدة أسئلة في رسالة واحدة، وقراءة "
            "الردود بصوت سوداني مستنسخ من التسجيلات الأصلية."
        )
    if re.match(r"^هل\s+(?:تستطيع|تقدر|يمكنك)\b", normalized):
        supported = (
            "تجيب",
            "الإجابة",
            "تبحث",
            "البحث",
            "تحلل",
            "تحليل",
            "تقارن",
            "مقارنة",
            "تحسب",
            "حساب",
            "تقرأ",
            "قراءة",
            "تتكلم",
            "تحدث",
            "صوت",
            "تاريخ",
            "ترجم",
        )
        if any(word in normalized for word in supported):
            capability = re.sub(
                r"^هل\s+(?:تستطيع|تقدر|يمكنك)\s+",
                "",
                normalized,
            ).strip()
            return (
                f"نعم، أستطيع {capability} ضمن المعلومات والأدوات المتاحة لي."
            )
        return (
            "لا أستطيع تأكيد قدرتي على ذلك بهذه الصياغة. وضّح المهمة المطلوبة "
            "وسأجيبك بنعم أو لا مباشرة."
        )
    return None


def exact_definition_answer(prompt):
    cleaned = prompt.strip(" ؟?!.،")
    match = re.match(
        r"^(?:ما\s+هو|ما\s+هي|ما\s+معنى|عرّف|عرف)\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    term = match.group(1).strip()
    local_definition = LOCAL_DEFINITIONS.get(normalized_term(term))
    if local_definition:
        return local_definition
    concept = resolve_concept(term)
    if not concept:
        return None
    return (
        f"{concept['summary']}\n\nالمصدر: {concept['source']}"
        "\nمرجع البيانات: تعريف مطابق للمصطلح"
    )


def seasonal_mountain_answer(prompt):
    normalized = prompt.strip().lower()
    match = re.match(
        r"^هل\s+يمكن\s+تسلق\s+(.+?)\s+في\s+موسم\s+(.+?)[؟?]?$",
        normalized,
    )
    if not match:
        return None
    mountain, season = match.groups()
    if normalized_term(mountain) in {"إيفرست", "ايفرست"} and season == "الصيف":
        concept = resolve_concept(mountain.strip())
        source = (
            concept["source"]
            if concept
            else "https://ar.wikipedia.org/wiki/جبل_إفرست"
        )
        return (
            "لا يُنصح عادةً بتسلق إيفرست في الصيف؛ لأن موسم الرياح الموسمية "
            "يجلب أمطارًا وثلوجًا كثيفة ويزيد خطر الانهيارات. الموسم الأشهر "
            "للتسلق هو الربيع، خصوصًا شهر مايو، وتوجد محاولات أقل في الخريف."
            f"\n\nالمصدر: {source}"
            "\nمرجع البيانات: معلومات الجبل وتحليل ظروف موسم التسلق"
        )
    concept = resolve_concept(mountain.strip())
    if not concept or not re.search(r"\b(?:جبل|قمة)\b", concept["summary"]):
        return None
    return (
        f"لا يمكن إعطاء جواب موثوق بنعم أو لا عن تسلق {mountain} في موسم "
        f"{season} دون بيانات الطقس، وحالة الطريق، وتصاريح الموسم."
    )


def bare_topic_answer(prompt):
    cleaned = prompt.strip(" ؟?!.،")
    words = cleaned.split()
    if not cleaned or len(words) > 4:
        return None
    if cleaned.startswith(QUESTION_STARTERS):
        return None
    if re.search(r"\b(?:هو|هي|كان|كانت|يكون|تستطيع|أريد|اريد|ابحث|قارن)\b", cleaned):
        return None
    concept = resolve_concept(cleaned)
    if not concept:
        return (
            f"هل تقصد موضوع «{cleaned}»؟ حدّد ما تريد معرفته عنه، "
            "مثل الموقع أو التعريف أو التاريخ."
        )
    return (
        f"{concept['summary']}\n\nالمصدر: {concept['source']}"
        "\nمرجع البيانات: تعريف مطابق للاسم"
    )


def generate_single(prompt, history=None):
    followup_answer = dialogue_followup_response(prompt, history or [])
    if followup_answer:
        return followup_answer
    dialogue_answer = dialogue_control_response(prompt)
    if dialogue_answer:
        return dialogue_answer
    contextual_prompt = resolve_follow_up(prompt, history or [])
    local_answer = exact_local_response(contextual_prompt)
    if local_answer:
        return local_answer
    request_answer = open_request_response(contextual_prompt)
    if request_answer:
        return request_answer
    meta_answer = assistant_meta_response(contextual_prompt)
    if meta_answer:
        return meta_answer
    definition_answer = exact_definition_answer(contextual_prompt)
    if definition_answer:
        return definition_answer
    seasonal_answer = seasonal_mountain_answer(contextual_prompt)
    if seasonal_answer:
        return seasonal_answer
    if difference_request(contextual_prompt):
        difference = difference_answer(contextual_prompt)
        return difference or (
            "تعذر التحقق من تعريفَي المصطلحين من المصادر الموثوقة الآن، لذلك "
            "لن أعرض مقارنة تخمينية أو نتيجة لا تتعلق بالسؤال."
        )
    ranked_request = ranked_cities_request(contextual_prompt)
    if ranked_request:
        ranked_answer = ranked_cities_answer(contextual_prompt)
        return ranked_answer or (
            "تعذر الوصول إلى بيانات المدن المهيكلة الآن، لذلك لن أعرض ترتيبًا "
            "تخمينيًا أو نتيجة غير مرتبطة بالسؤال."
        )
    compared_answer = comparison_answer(contextual_prompt)
    if compared_answer:
        return compared_answer
    calculated_answer = math_response(contextual_prompt)
    if calculated_answer:
        return calculated_answer
    live_answer = temporal_response(contextual_prompt)
    if live_answer:
        return live_answer
    topic_answer = bare_topic_answer(contextual_prompt)
    if topic_answer:
        return topic_answer
    domain = question_domain(contextual_prompt)
    normalized = contextual_prompt.lower()
    internet_first = (
        domain in {"dictionary", "medical", "technology", "economy", "science"}
        or wikidata_intent(contextual_prompt) is not None
        or any(
            keyword in normalized
            for keyword in ("أحدث", "احدث", "آخر إصدار", "اخر اصدار", "مؤسس")
        )
    )
    if internet_first:
        return search_internet(contextual_prompt)
    local_answer = retrieve_response(contextual_prompt)
    if local_answer:
        return local_answer
    return search_internet(contextual_prompt)


def answer_topic(question):
    topic = question.strip(" ؟?")
    topic = re.sub(
        r"^(?:ما هي|ما هو|ما|ماذا|من هو|من هي|من|متى|أين|اين|كيف|كم|هل)\s+",
        "",
        topic,
        flags=re.IGNORECASE,
    )
    return topic[:80].strip() or "هذا السؤال"


def reinforce_question_terms(question, answer):
    if not answer or answer.startswith(
        ("مرحب", "أهلًا", "هلا", "وعليكم", "صباح", "مساء")
    ):
        return answer
    question_words = normalize_words(question)
    answer_without_sources = answer.split("\n\nالمصدر:", 1)[0]
    if question_words & normalize_words(answer_without_sources):
        return answer
    topic = answer_topic(question)
    if topic == "هذا السؤال":
        return answer
    if answer.startswith("نعم،"):
        return f"نعم، وبخصوص {topic}: {answer[5:].lstrip()}"
    if answer.startswith("لا،"):
        return f"لا، وبخصوص {topic}: {answer[3:].lstrip()}"
    return answer


def professionalize_answer(question, answer, transition=None):
    if answer.startswith(("بالنسبة إلى", "بالنسبة لـ", "أما بخصوص")):
        return answer
    if difference_request(question) or comparison_request(question):
        return answer
    if scientific_search_request(question):
        return answer
    topic = answer_topic(question)
    normalized = question.strip().lower()
    if normalized.startswith(("ما هو", "ما هي", "ما معنى", "عرّف", "عرف")):
        return answer
    if normalized.startswith("هل ") and answer.startswith(("نعم", "لا")):
        return answer
    prefix = transition or f"عن {topic}:"
    return f"{prefix}\n{answer}"


def generate(
    prompt,
    max_chars=100,
    use_neural_generation=False,
    history=None,
    response_mode=None,
):
    if not use_neural_generation:
        preference = response_detail_preference(prompt, history or []) or (
            response_mode
            if response_mode in {"brief", "balanced", "detailed"}
            else "balanced"
        )
        meta_answer = assistant_meta_response(prompt)
        if meta_answer:
            return to_sudanese_text(
                apply_response_preference(meta_answer, preference)
            )
        questions = split_questions(prompt)
        if len(questions) <= 1:
            answer = generate_single(prompt, history)
            answer = apply_question_response_preference(
                prompt, answer, preference
            )
            answer = reinforce_question_terms(prompt, answer)
            normalized = prompt.strip().lower()
            if dialogue_control_response(prompt) or dialogue_followup_response(
                prompt, history or []
            ):
                return to_sudanese_text(answer)
            if normalized.startswith("هل ") and answer.startswith(
                ("نعم", "لا", "لا أستطيع", "لا يمكن")
            ):
                return to_sudanese_text(answer)
            if answer.startswith(
                (
                    "أستطيع ",
                    "بالتأكيد",
                    "مرحب",
                    "أهلًا",
                    "هلا",
                    "وعليكم",
                    "صباح",
                    "مساء",
                    "هل تقصد ",
                    "وضّح ",
                    "حدد ",
                    "حدّد ",
                )
            ):
                return to_sudanese_text(answer)
            if any(
                greeting in normalized
                for greeting in (
                    "مرحبا",
                    "مرحبًا",
                    "السلام عليكم",
                    "أهلا",
                    "اهلا",
                    "هلا",
                    "هاي",
                    "hello",
                )
            ):
                return to_sudanese_text(answer)
            return to_sudanese_text(professionalize_answer(prompt, answer))

        answers = []
        transitions = (
            "بالنسبة إلى {topic}، فالإجابة هي:",
            "أما بخصوص {topic}، فالإجابة هي:",
            "وبالانتقال إلى {topic}، فالإجابة هي:",
            "وفيما يتعلق بـ {topic}، فالنتيجة هي:",
        )
        for index, question in enumerate(questions, start=1):
            answer = generate_single(question, history)
            answer = apply_question_response_preference(
                question, answer, preference
            )
            answer = reinforce_question_terms(question, answer)
            topic = answer_topic(question)
            transition = transitions[min(index - 1, len(transitions) - 1)].format(
                topic=topic
            )
            answers.append(professionalize_answer(question, answer, transition))
        return to_sudanese_text("\n\n".join(answers))

    import torch

    model, char_to_id = load_model()
    id_to_char = {index: char for char, index in char_to_id.items()}
    prefix = f"المستخدم: {prompt}\nالمساعد:"
    known_prefix = "".join(char for char in prefix if char in char_to_id)
    tokens = torch.tensor([[char_to_id[char] for char in known_prefix]])
    input_ids = [char_to_id[char] for char in known_prefix]
    tokens = torch.tensor([input_ids])

    with torch.no_grad():
        logits, hidden = model(tokens)
        next_id = int(logits[0, -1].argmax())
        output = []
        for _ in range(max_chars):
            logits, _ = model(tokens)
            next_id = int(logits[0, -1].argmax())
            char = id_to_char[next_id]
            if char == "§":
                break
            output.append(char)
            token = torch.tensor([[next_id]])
            logits, hidden = model(token, hidden)
            next_id = int(logits[0, -1].argmax())
            tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)
            if tokens.size(1) > 500: break

    return to_sudanese_text("".join(output).strip())


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "مرحبا"
    print(generate(prompt))
