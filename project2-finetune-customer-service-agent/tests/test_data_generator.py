"""
测试数据生成器
"""

from src.data.data_generator import CustomerServiceDataGenerator, INTENT_TEMPLATES


class TestDataGenerator:
    def setup_method(self):
        self.gen = CustomerServiceDataGenerator()

    def test_generate_count(self):
        samples = self.gen.generate(300)
        assert len(samples) == 300

    def test_intents_covered(self):
        samples = self.gen.generate(500)
        intents = {s.intent for s in samples}
        for intent in INTENT_TEMPLATES:
            assert intent in intents, f"Missing intent: {intent}"

    def test_adversarial_count(self):
        samples = self.gen.generate(500)
        adv = [s for s in samples if s.is_adversarial]
        assert len(adv) > 0

    def test_dpo_pairs_format(self):
        samples = self.gen.generate(100)
        pairs = self.gen.generate_dpo_pairs(samples, 50)
        assert len(pairs) == 50
        for p in pairs:
            assert "chosen" in p and "rejected" in p
            assert p["chosen"] != p["rejected"]
