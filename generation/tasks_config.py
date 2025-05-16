
# tasks_config.py

TASKS = {
    "T1": {
        "name": "Sentence Context Anomaly",
        "type": "index",
        "topics": ["philosophy", "society", "psychology"],
        "style": ["GRE"],
        "factors": ["minor topic shift", "semantic deviation"],
        "example": {
            "context": [
                "Utilitarianism, as articulated by Jeremy Bentham, evaluates actions based on their capacity to produce pleasure and minimize pain.",
                "John Stuart Mill refined this theory by distinguishing between higher and lower forms of pleasure.",
                "Critics have challenged utilitarianism for failing to account for justice or rights in its calculus of utility.",
                "Nonetheless, the theory remains influential in contemporary policy-making, especially in public health and welfare economics.",
                "Some argue that utilitarianism justifies deception in all cases, as long as it produces personal satisfaction."
            ],
            "anomaly_index": 4
        }
    },
    "T2": {
        "name": "Paragraph Order Consistency",
        "type": "binary",
        "topics": ["science", "economics", "politics"],
        "style": ["LSAT", "GMAT"],
        "factors": ["sentence reordering"],
        "example": {
            "context": [
                "The expansion of urban green spaces has been linked to lower stress levels and improved mental well-being.",
                "However, the distribution of such spaces often favors wealthier neighborhoods.",
                "Urban planners are increasingly interested in how access to nature affects long-term public health outcomes.",
                "This raises equity concerns, particularly for marginalized communities with limited access to parks or tree cover.",
                "As a result, some cities have adopted policies that prioritize green space development in underserved areas."
            ],
            "is_coherent": False
        }
    },
    "T3": {
        "name": "Blank-based Choice Anomaly",
        "type": "index",
        "topics": ["literature", "psychology", "philosophy"],
        "style": ["GRE"],
        "factors": ["lexical fit", "collocation"],
        "example": {
            "sentence": "In the early 2000s, the widespread adoption of _____ began to reshape how people interacted socially, particularly among young adults.",
            "choices": [
                "online forums",
                "instant messaging platforms",
                "MySpace",
                "mobile dating apps",
                "blogging communities"
            ],
            "anomaly_index": 3
        }
    },
    "T4": {
        "name": "Bridge Sentence Evaluation",
        "type": "index",
        "topics": ["economics", "society", "policy"],
        "style": ["GMAT", "LSAT"],
        "factors": ["weak logical connection", "abrupt topic shift"],
        "example": {
            "paragraph_1": [
                "Cryptocurrencies have rapidly emerged as an alternative to traditional fiat currencies, attracting both retail investors and institutional interest.",
                "Despite their promise, concerns over volatility and lack of regulation remain significant barriers to widespread adoption."
            ],
            "paragraph_2": [
                "Central banks around the world are now exploring the issuance of digital currencies to modernize payment systems and retain monetary control.",
                "These central bank digital currencies (CBDCs) aim to offer the convenience of crypto while maintaining the stability and trust of government-issued money."
            ],
            "bridges": [
                "CBDCs are essentially government-sanctioned counterparts to decentralized cryptocurrencies.",
                "This shift from cash to digital forms reflects an ongoing trend toward financial innovation.",
                "While crypto investors seek returns, central banks prioritize macroeconomic stability.",
                "The increasing use of digital wallets has made the user experience of CBDCs nearly indistinguishable from that of crypto.",
                "Smart contracts enable automated transactions without third-party intermediaries, revolutionizing how finance operates."
            ],
            "anomaly_index": 4
        }
    },
    "T5": {
        "name": "Referential Ambiguity",
        "type": "index",
        "topics": ["psychology", "literature", "philosophy"],
        "style": ["GRE"],
        "factors": ["ambiguous pronouns", "unclear referents"],
        "example": {
            "context": [
                "Descartes' dualism posits a clear distinction between mind and body, each governed by different principles.",
                "He argued that while the body operates in the physical world, the mind is non-material and indivisible.",
                "Spinoza rejected this view, asserting that both are attributes of a single substance.",
                "Although he elaborated this idea in his Ethics, its reception was mixed due to its abstract nature.",
                "This led many to question whether he was opposing Descartes or merely reframing it."
            ],
            "anomaly_index": 4
        }
    },
    "T6": {
        "name": "Logical Contradiction",
        "type": "index",
        "topics": ["science", "economics", "politics"],
        "style": ["LSAT", "GMAT"],
        "factors": ["contradictory claims", "causal reversal"],
        "example": {
            "context": [
                "A recent study found that people who consume diets rich in plant-based foods have lower rates of heart disease.",
                "Researchers also noted that moderate red wine consumption may offer cardiovascular benefits.",
                "Other studies suggest that regular exercise improves heart health across all age groups.",
                "However, one clinical trial concluded that individuals with the highest plant-based intake had significantly higher cholesterol levels.",
                "Despite some debate, most medical professionals recommend a combination of diet and exercise for heart health."
            ],
            "anomaly_index": 3
        }
    },
    "T7": {
        "name": "Tone / Style Violation",
        "type": "index",
        "topics": ["literature", "philosophy"],
        "style": ["GRE"],
        "factors": ["tone shift", "register mismatch"],
        "example": {
            "context": [
                "Nietzsche’s critique of morality centers on the idea that traditional ethical systems are rooted in ressentiment.",
                "He distinguishes between master and slave morality, the latter arising from the inversion of noble values.",
                "Rather than promoting strength, modern morality, in his view, elevates weakness and conformity.",
                "This analysis has had lasting impact on existential and postmodern thought.",
                "It’s kind of like when people pretend to be nice just to look good—totally fake."
            ],
            "anomaly_index": 4
        }
    }
}
