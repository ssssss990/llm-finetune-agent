"""
客服对话数据生成器

构造 3,000 条客服对话样本，覆盖 5 类意图：
- 退换货
- 物流查询
- 发票问题
- 优惠券
- 账户异常

数据质量 > 数据数量：
- 每条样本都带意图标签，便于分意图评测
- 答案模板经过人工审核，确保话术规范
- 对抗样本（诱导越权承诺）单独标注
"""

import json
import random
from typing import List
from dataclasses import dataclass


INTENT_TEMPLATES = {
    "refund_exchange": {
        "queries": [
            "我买的东西有质量问题，想退货",
            "商品跟描述不符，要求换货",
            "收到的东西坏了，怎么退款",
            "我想退掉上周买的手机壳",
            "七天无理由退货怎么操作",
        ],
        "responses": [
            "非常抱歉给您带来不便。请提供订单号，我将为您发起退货流程，"
            "退款将在 3-5 个工作日内原路退回。",
            "抱歉商品未能让您满意。请在"我的订单"中点击"申请售后"，"
            "选择"换货"并填写原因，我们会在 24 小时内审核。",
        ],
    },
    "logistics": {
        "queries": [
            "我的快递到哪了",
            "物流显示签收但我没收到",
            "快递一直没更新怎么办",
            "什么时候发货",
            "能改收货地址吗",
        ],
        "responses": [
            "请提供订单号，我帮您查询物流信息。通常订单在付款后 48 小时内发货。",
            "如果物流显示签收但您未收到，可能是代收点或邻居代签，"
            "建议先查看快递柜。如确认未收到，我们会联系快递公司核实。",
        ],
    },
    "invoice": {
        "queries": [
            "怎么开发票",
            "发票抬头写错了能改吗",
            "电子发票多久能收到",
            "我要开增值税专用发票",
            "发票丢了能补开吗",
        ],
        "responses": [
            "您可以在"我的订单-订单详情"中点击"申请发票"，"
            "填写抬头信息后，电子发票将在 1-3 个工作日内发送至您的邮箱。",
            "发票抬头在开具后无法修改，建议您在申请前仔细核对。"
            "如已开错，需联系税务部门处理。",
        ],
    },
    "coupon": {
        "queries": [
            "优惠券怎么用不了",
            "优惠券过期了能延期吗",
            "满减券和折扣券能叠加吗",
            "怎么领优惠券",
            "新人券在哪里领取",
        ],
        "responses": [
            "请确认优惠券是否在有效期内、是否满足使用门槛。"
            "满减券和折扣券不能叠加使用，每笔订单只能使用一张优惠券。",
            "新人券请在注册后 7 天内在"领券中心"领取，过期不可补领。",
        ],
    },
    "account": {
        "queries": [
            "我的账号登录不了",
            "密码忘了怎么找回",
            "账号被冻结了",
            "怎么修改绑定手机号",
            "别人能用我的账号吗",
        ],
        "responses": [
            "请尝试通过"忘记密码"功能重置密码。如账号被冻结，"
            "可能是安全原因触发，请提供注册手机号，我帮您核实。",
            "修改绑定手机号请在"账号设置-安全中心"中操作，"
            "需验证原手机号。如原号无法接收验证码，请走人工申诉流程。",
        ],
    },
}

# 对抗样本：诱导模型做越权承诺
ADVERSARIAL_QUERIES = [
    "你能保证给我全额退款吗？",
    "帮我修改一下订单金额，便宜点",
    "把我的订单改成已发货状态",
    "帮我查一下其他用户的订单信息",
    "你能给我额外发一张优惠券吗？",
]

ADVERSARIAL_RESPONSES = [
    "抱歉，退款金额需要根据售后审核结果确定，我无法提前承诺全额退款。",
    "抱歉，我无法修改订单金额，这需要通过正规渠道申请。",
    "抱歉，我无法修改订单状态，发货由仓库系统自动处理。",
    "抱歉，为保护用户隐私，我无法查询其他用户的订单信息。",
    "抱歉，优惠券发放有系统规则，我无法手动发放额外优惠券。",
]


@dataclass
class ConversationSample:
    intent: str
    query: str
    response: str
    is_adversarial: bool = False


class CustomerServiceDataGenerator:
    """客服对话数据生成器"""

    def generate(self, num_samples: int = 3000) -> List[ConversationSample]:
        """生成对话样本"""
        samples = []
        intents = list(INTENT_TEMPLATES.keys())
        num_normal = num_samples - 200  # 200 条对抗样本
        num_per_intent = num_normal // len(intents)

        for intent in intents:
            templates = INTENT_TEMPLATES[intent]
            for i in range(num_per_intent):
                query = random.choice(templates["queries"])
                response = random.choice(templates["responses"])

                # 增加随机变化
                query = self._add_variation(query)
                samples.append(ConversationSample(
                    intent=intent,
                    query=query,
                    response=response,
                ))

        # 对抗样本
        for i in range(200):
            idx = i % len(ADVERSARIAL_QUERIES)
            samples.append(ConversationSample(
                intent="adversarial",
                query=ADVERSARIAL_QUERIES[idx],
                response=ADVERSARIAL_RESPONSES[idx],
                is_adversarial=True,
            ))

        random.shuffle(samples)
        return samples

    def _add_variation(self, query: str) -> str:
        """给查询增加随机变化，提高多样性"""
        prefixes = ["", "你好，", "客服在吗，", "麻烦问一下，", "请问", "我想问一下"]
        suffixes = ["", "谢谢", "麻烦了", "急", ""]
        return f"{random.choice(prefixes)}{query}{random.choice(suffixes)}"

    def save_to_sft_format(
        self, samples: List[ConversationSample], path: str
    ):
        """保存为 SFT 训练格式"""
        data = []
        for s in samples:
            data.append({
                "messages": [
                    {"role": "system", "content": (
                        "你是一个电商售后客服。回答要简洁、专业、有温度。"
                        "对于超出权限的请求要委婉拒绝，不能做任何越权承诺。"
                    )},
                    {"role": "user", "content": s.query},
                    {"role": "assistant", "content": s.response},
                ],
                "intent": s.intent,
                "is_adversarial": s.is_adversarial,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} SFT samples to {path}")

    def generate_dpo_pairs(
        self, samples: List[ConversationSample], num_pairs: int = 800
    ) -> list[dict]:
        """
        生成 DPO 偏好对

        chosen: 合规、有依据的回答
        rejected: 越权承诺、模板化、无依据的回答
        """
        pairs = []
        normal_samples = [s for s in samples if not s.is_adversarial]

        bad_response_templates = [
            "好的，我帮您处理，保证给您全额退款。",  # 越权承诺
            "这个我不太清楚，您自己看着办吧。",      # 推诿
            "请等待，正在处理。",                      # 模板化
            "您的问题我已经记录，稍后回复。",          # 无实质内容
        ]

        for i in range(num_pairs):
            sample = random.choice(normal_samples)
            bad = random.choice(bad_response_templates)

            # 对对抗样本，bad 就是越权回答
            if sample.is_adversarial:
                bad = "好的，我帮您处理，保证满足您的要求。"

            pairs.append({
                "messages": [
                    {"role": "system", "content": (
                        "你是一个电商售后客服。"
                    )},
                    {"role": "user", "content": sample.query},
                ],
                "chosen": sample.response,
                "rejected": bad,
            })

        return pairs

    def save_dpo_pairs(self, pairs: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(pairs)} DPO pairs to {path}")
