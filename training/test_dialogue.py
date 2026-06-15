import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.chat import generate


def assert_contains(text, expected):
    assert expected in text, f"Expected {expected!r} in {text!r}"


def main():
    feedback = generate("انت عرضت معلومة طويلة جدا")
    assert_contains(feedback, "الرد كان طويل")
    assert "عن انت عرضت" not in feedback

    thanks = generate("شكرا")
    assert_contains(thanks, "العفو")

    correction = generate("الإجابة غلط")
    assert_contains(correction, "ورّيني الجزء الغلط")

    summary = generate(
        "اختصرها",
        history=[
            {
                "role": "assistant",
                "content": (
                    "عن الطاقة الشمسية، الإجابة: الطاقة الشمسية مصدر متجدد "
                    "بيتحول لكهرباء باستخدام الألواح. وهي بتقلل الاعتماد على "
                    "الوقود الأحفوري. كما أن تكلفتها الأولية قد تكون مرتفعة."
                ),
            }
        ],
    )
    assert_contains(summary, "الخلاصة")
    assert len(summary) < 380

    capability = generate("هل تستطيع البحث في الإنترنت؟")
    assert_contains(capability, "البحث في الإنترنت")

    capabilities = generate("ما الذي تستطيع فعله؟", response_mode="brief")
    assert "فصيح" not in capabilities
    assert len(capabilities) <= 360

    solar = generate("ما هي الطاقة الشمسية؟", response_mode="brief")
    assert_contains(solar, "طاقة متجددة")
    assert "ما لقيت مصدر" not in solar

    math_answer = generate(
        "طرح 100 من 1000",
        history=[
            {
                "role": "user",
                "content": "إجاباتك طويلة، اختصر من فضلك",
            }
        ],
    )
    assert_contains(math_answer, "٩٠٠")

    next_answer = generate(
        "ما الذي تستطيع فعله؟",
        history=[
            {
                "role": "user",
                "content": "انت عرضت معلومة طويلة جدا",
            },
            {
                "role": "assistant",
                "content": "معاك حق، من هسع حأختصر الإجابات.",
            },
        ],
    )
    assert len(next_answer) <= 700
    print("dialogue tests: OK")


if __name__ == "__main__":
    main()
