"""Fixed prompt set v1 (2026-09-16). Domains chosen to probe domain clustering of experts.
Written for this project; not taken from any dataset."""

PROMPTS = [
    # code
    ("code", "Write a Python function that parses a CSV file of transactions and returns the total amount per customer, with type hints and error handling."),
    ("code", "Implement a thread-safe LRU cache in Rust and explain the ownership decisions."),
    ("code", "Write a JavaScript debounce function and a unit test for it using Jest."),
    ("code", "Explain what this SQL does and optimize it: SELECT * FROM orders o WHERE o.customer_id IN (SELECT id FROM customers WHERE country = 'UZ') ORDER BY o.created_at DESC;"),
    # english technical
    ("en_tech", "Explain how TCP congestion control works, including slow start and fast recovery."),
    ("en_tech", "Describe the trade-offs between mixture-of-experts and dense transformer models for inference serving."),
    ("en_tech", "How does a Kubernetes scheduler decide where to place a pod? Cover filtering and scoring."),
    # english prose
    ("en_prose", "Write a short story about a lighthouse keeper who discovers a message in a bottle."),
    ("en_prose", "Describe a busy morning market in Samarkand from the point of view of a spice seller."),
    ("en_prose", "Write a heartfelt letter from a grandmother to her grandson who is leaving for university."),
    # russian
    ("ru", "Напиши короткий рассказ о студенте, который впервые приехал в большой город."),
    ("ru", "Объясни простыми словами, как работает электронная почта."),
    ("ru", "Какие плюсы и минусы у удалённой работы? Ответь развёрнуто."),
    # uzbek
    ("uz", "O'zbekiston tarixidagi Buyuk Ipak yo'lining ahamiyati haqida qisqacha yozib bering."),
    ("uz", "Yangi boshlovchi dasturchi uchun Python tilini o'rganish rejasini tuzing."),
    # chinese
    ("zh", "请用通俗的语言解释什么是区块链技术。"),
    ("zh", "写一首关于秋天的现代诗。"),
    # math
    ("math", "Solve step by step: a train leaves at 9:00 at 80 km/h, another leaves the same station at 10:30 at 120 km/h. When does the second catch up?"),
    ("math", "Prove that the sum of the first n odd numbers equals n squared."),
    ("math", "Find all real x such that x^3 - 6x^2 + 11x - 6 = 0, showing your reasoning."),
    # dialogue / assistant
    ("chat", "I have a job interview tomorrow for a data analyst role. Can you help me prepare?"),
    ("chat", "Plan a 3-day trip to Istanbul on a moderate budget."),
    ("chat", "My houseplant leaves are turning yellow. What could be wrong?"),
    ("chat", "Give me a weekly workout plan for a beginner with no equipment."),
]
