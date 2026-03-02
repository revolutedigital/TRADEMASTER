"""NLP-based market analysis for cryptocurrency news and social media."""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class SentimentLabel(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class NLPResult:
    """Result of NLP analysis."""
    text: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: SentimentLabel
    confidence: float
    entities: list[dict]
    topics: list[str]
    key_phrases: list[str]
    urgency: float  # 0.0 to 1.0


# Crypto-specific sentiment lexicon
CRYPTO_LEXICON = {
    # Very bullish terms
    "moon": 0.8, "moonshot": 0.9, "bullish": 0.7, "breakout": 0.6,
    "all-time high": 0.8, "ath": 0.8, "pump": 0.6, "rally": 0.7,
    "adoption": 0.5, "institutional": 0.5, "accumulate": 0.6,
    "hodl": 0.5, "diamond hands": 0.6, "to the moon": 0.8,
    "green candle": 0.5, "golden cross": 0.7, "bull run": 0.8,
    "halving": 0.5, "etf approval": 0.8, "etf approved": 0.9,
    "mass adoption": 0.7, "bullish divergence": 0.6, "oversold": 0.4,

    # Bullish terms
    "buy": 0.3, "long": 0.3, "support": 0.2, "bounce": 0.3,
    "upgrade": 0.4, "partnership": 0.4, "launch": 0.3,
    "growth": 0.3, "innovation": 0.3, "milestone": 0.4,

    # Bearish terms
    "dump": -0.6, "crash": -0.7, "bear": -0.5, "bearish": -0.7,
    "sell": -0.3, "short": -0.3, "resistance": -0.2,
    "correction": -0.4, "decline": -0.4, "drop": -0.4,
    "death cross": -0.7, "red candle": -0.4, "overbought": -0.4,
    "bearish divergence": -0.6,

    # Very bearish terms
    "rugpull": -0.9, "rug pull": -0.9, "scam": -0.8, "hack": -0.8,
    "exploit": -0.7, "bankrupt": -0.9, "bankruptcy": -0.9,
    "collapse": -0.8, "capitulation": -0.7, "liquidation": -0.6,
    "panic": -0.6, "fud": -0.5, "ponzi": -0.9,
    "delisting": -0.7, "ban": -0.6, "regulation crackdown": -0.7,
    "paper hands": -0.3, "rekt": -0.6,

    # Neutral/contextual
    "blockchain": 0.1, "defi": 0.1, "nft": 0.0, "web3": 0.1,
    "staking": 0.2, "yield": 0.1, "airdrop": 0.2,
    "whale": 0.0, "volume": 0.0, "market cap": 0.0,
}

# Named entities for crypto
CRYPTO_ENTITIES = {
    "btc": "Bitcoin", "bitcoin": "Bitcoin",
    "eth": "Ethereum", "ethereum": "Ethereum",
    "sol": "Solana", "solana": "Solana",
    "bnb": "BNB", "xrp": "XRP", "ripple": "XRP",
    "ada": "Cardano", "cardano": "Cardano",
    "doge": "Dogecoin", "dogecoin": "Dogecoin",
    "avax": "Avalanche", "dot": "Polkadot",
    "matic": "Polygon", "polygon": "Polygon",
    "link": "Chainlink", "chainlink": "Chainlink",
    "uni": "Uniswap", "uniswap": "Uniswap",
    "aave": "Aave", "usdt": "Tether", "usdc": "USD Coin",
    # Exchanges
    "binance": "Binance", "coinbase": "Coinbase",
    "kraken": "Kraken", "ftx": "FTX",
    # Institutions
    "sec": "SEC", "fed": "Federal Reserve",
    "blackrock": "BlackRock", "grayscale": "Grayscale",
    "microstrategy": "MicroStrategy",
}

# Urgency indicators
URGENCY_TERMS = {
    "breaking": 0.9, "urgent": 0.8, "alert": 0.7,
    "just in": 0.8, "now": 0.3, "imminent": 0.7,
    "flash": 0.8, "emergency": 0.9, "critical": 0.8,
    "warning": 0.6, "live": 0.4,
}


class CryptoNLPAnalyzer:
    """
    NLP analyzer specialized for cryptocurrency market text.

    Features:
    - Crypto-specific sentiment lexicon
    - Named entity recognition for assets, exchanges, institutions
    - Topic extraction and classification
    - Urgency detection
    - Aggregated multi-source sentiment
    """

    def __init__(self):
        self.lexicon = CRYPTO_LEXICON
        self.entities = CRYPTO_ENTITIES
        self.urgency_terms = URGENCY_TERMS
        logger.info("nlp_analyzer_initialized", lexicon_size=len(self.lexicon),
                    entity_count=len(self.entities))

    def analyze(self, text: str) -> NLPResult:
        """Analyze a single text for sentiment, entities, and topics."""
        text_lower = text.lower()
        tokens = self._tokenize(text_lower)

        # Sentiment analysis
        sentiment_score, contributing_terms = self._compute_sentiment(tokens, text_lower)
        confidence = self._compute_confidence(contributing_terms, tokens)

        # Named entity recognition
        entities = self._extract_entities(text_lower)

        # Topic extraction
        topics = self._extract_topics(tokens, text_lower)

        # Key phrases
        key_phrases = self._extract_key_phrases(text, contributing_terms)

        # Urgency detection
        urgency = self._compute_urgency(text_lower)

        # Determine label
        sentiment_label = self._score_to_label(sentiment_score)

        return NLPResult(
            text=text[:500],
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            entities=entities,
            topics=topics,
            key_phrases=key_phrases,
            urgency=urgency,
        )

    def analyze_batch(self, texts: list[str]) -> dict:
        """Analyze multiple texts and aggregate results."""
        results = [self.analyze(text) for text in texts]

        if not results:
            return {
                "overall_sentiment": 0.0,
                "overall_label": SentimentLabel.NEUTRAL.value,
                "confidence": 0.0,
                "n_analyzed": 0,
                "sentiment_distribution": {},
                "top_entities": [],
                "top_topics": [],
            }

        scores = [r.sentiment_score for r in results]
        confidences = [r.confidence for r in results]

        # Weighted average by confidence
        if sum(confidences) > 0:
            weighted_sentiment = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
        else:
            weighted_sentiment = float(np.mean(scores))

        # Sentiment distribution
        dist = {}
        for r in results:
            label = r.sentiment_label.value
            dist[label] = dist.get(label, 0) + 1

        # Top entities
        entity_counts: dict[str, int] = {}
        for r in results:
            for entity in r.entities:
                name = entity["name"]
                entity_counts[name] = entity_counts.get(name, 0) + 1
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top topics
        topic_counts: dict[str, int] = {}
        for r in results:
            for topic in r.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Max urgency
        max_urgency = max(r.urgency for r in results)

        return {
            "overall_sentiment": float(weighted_sentiment),
            "overall_label": self._score_to_label(weighted_sentiment).value,
            "confidence": float(np.mean(confidences)),
            "n_analyzed": len(results),
            "sentiment_distribution": dist,
            "top_entities": [{"name": name, "count": count} for name, count in top_entities],
            "top_topics": [{"topic": topic, "count": count} for topic, count in top_topics],
            "max_urgency": float(max_urgency),
            "individual_results": [
                {
                    "text_preview": r.text[:100],
                    "sentiment": r.sentiment_score,
                    "label": r.sentiment_label.value,
                    "confidence": r.confidence,
                }
                for r in results[:20]  # Limit to 20 for response size
            ],
        }

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization with crypto-aware handling."""
        text = re.sub(r'[^\w\s\-$#@]', ' ', text)
        tokens = text.split()
        return [t.strip() for t in tokens if len(t.strip()) > 1]

    def _compute_sentiment(self, tokens: list[str], full_text: str) -> tuple[float, list[tuple[str, float]]]:
        """Compute sentiment score using crypto lexicon."""
        contributing_terms = []
        total_score = 0.0
        n_matches = 0

        # Check individual tokens
        for token in tokens:
            if token in self.lexicon:
                score = self.lexicon[token]
                contributing_terms.append((token, score))
                total_score += score
                n_matches += 1

        # Check multi-word phrases
        for phrase, score in self.lexicon.items():
            if ' ' in phrase and phrase in full_text:
                contributing_terms.append((phrase, score))
                total_score += score
                n_matches += 1

        # Handle negation
        negation_words = {"not", "no", "never", "don't", "doesn't", "isn't", "won't", "can't", "shouldn't"}
        for i, token in enumerate(tokens):
            if token in negation_words and i + 1 < len(tokens) and tokens[i + 1] in self.lexicon:
                # Reverse the next term's sentiment
                negated_score = -self.lexicon[tokens[i + 1]]
                contributing_terms.append((f"NOT {tokens[i + 1]}", negated_score))
                total_score += negated_score * 2  # Double weight for negation correction
                n_matches += 1

        # Normalize to [-1, 1]
        if n_matches > 0:
            avg_score = total_score / n_matches
            normalized = max(-1.0, min(1.0, avg_score))
        else:
            normalized = 0.0

        return normalized, contributing_terms

    def _compute_confidence(self, contributing_terms: list[tuple[str, float]],
                           tokens: list[str]) -> float:
        """Compute confidence based on evidence strength."""
        if not tokens:
            return 0.0

        # Coverage: what fraction of tokens are sentiment-bearing
        coverage = len(contributing_terms) / max(len(tokens), 1)

        # Agreement: do all terms point in the same direction
        if contributing_terms:
            signs = [1 if score > 0 else -1 for _, score in contributing_terms if score != 0]
            if signs:
                agreement = abs(sum(signs)) / len(signs)
            else:
                agreement = 0.5
        else:
            agreement = 0.0

        # Strength: average absolute score
        if contributing_terms:
            strength = np.mean([abs(score) for _, score in contributing_terms])
        else:
            strength = 0.0

        confidence = (coverage * 0.3 + agreement * 0.4 + strength * 0.3)
        return min(1.0, float(confidence))

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text."""
        found = []
        seen = set()

        for keyword, entity_name in self.entities.items():
            if keyword in text and entity_name not in seen:
                # Determine entity type
                if entity_name in ("SEC", "Federal Reserve"):
                    entity_type = "REGULATOR"
                elif entity_name in ("Binance", "Coinbase", "Kraken", "FTX"):
                    entity_type = "EXCHANGE"
                elif entity_name in ("BlackRock", "Grayscale", "MicroStrategy"):
                    entity_type = "INSTITUTION"
                elif entity_name in ("Tether", "USD Coin"):
                    entity_type = "STABLECOIN"
                else:
                    entity_type = "CRYPTOCURRENCY"

                found.append({
                    "name": entity_name,
                    "type": entity_type,
                    "keyword": keyword,
                })
                seen.add(entity_name)

        return found

    def _extract_topics(self, tokens: list[str], text: str) -> list[str]:
        """Extract topics from text."""
        topic_keywords = {
            "regulation": ["regulation", "sec", "ban", "compliance", "law", "legal", "policy"],
            "defi": ["defi", "yield", "liquidity", "pool", "staking", "farming"],
            "nft": ["nft", "nfts", "collectible", "metaverse", "opensea"],
            "mining": ["mining", "miner", "hashrate", "pow", "proof of work"],
            "trading": ["trade", "trading", "buy", "sell", "long", "short", "leverage"],
            "technology": ["upgrade", "fork", "mainnet", "testnet", "protocol", "layer 2", "l2"],
            "institutional": ["institutional", "etf", "fund", "blackrock", "grayscale"],
            "security": ["hack", "exploit", "vulnerability", "audit", "security"],
            "macro": ["fed", "inflation", "interest rate", "recession", "gdp"],
            "adoption": ["adoption", "payment", "merchant", "integration"],
        }

        found = []
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found.append(topic)
                    break

        return list(set(found))

    def _extract_key_phrases(self, text: str,
                             contributing_terms: list[tuple[str, float]]) -> list[str]:
        """Extract key phrases from text."""
        phrases = []

        # Add high-impact sentiment terms
        sorted_terms = sorted(contributing_terms, key=lambda x: abs(x[1]), reverse=True)
        for term, score in sorted_terms[:5]:
            phrases.append(term)

        # Extract quoted phrases
        quotes = re.findall(r'"([^"]+)"', text)
        phrases.extend(quotes[:3])

        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text)
        phrases.extend(hashtags[:3])

        return list(set(phrases))[:10]

    def _compute_urgency(self, text: str) -> float:
        """Compute urgency score from text."""
        max_urgency = 0.0
        for term, score in self.urgency_terms.items():
            if term in text:
                max_urgency = max(max_urgency, score)

        # Exclamation marks increase urgency
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            max_urgency = min(1.0, max_urgency + exclamation_count * 0.05)

        # ALL CAPS words increase urgency
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if len(caps_words) > 2:
            max_urgency = min(1.0, max_urgency + 0.1)

        return max_urgency

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label."""
        if score >= 0.5:
            return SentimentLabel.VERY_BULLISH
        elif score >= 0.15:
            return SentimentLabel.BULLISH
        elif score > -0.15:
            return SentimentLabel.NEUTRAL
        elif score > -0.5:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.VERY_BEARISH

    def get_market_pulse(self, texts: list[str]) -> dict:
        """Get a quick market pulse from a batch of texts."""
        analysis = self.analyze_batch(texts)
        score = analysis["overall_sentiment"]

        # Trading signal suggestion based on NLP
        if score >= 0.4 and analysis["confidence"] > 0.5:
            signal = "STRONG_BUY"
        elif score >= 0.15:
            signal = "BUY"
        elif score > -0.15:
            signal = "HOLD"
        elif score > -0.4:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"

        return {
            "signal": signal,
            "sentiment_score": score,
            "confidence": analysis["confidence"],
            "urgency": analysis.get("max_urgency", 0.0),
            "dominant_topics": [t["topic"] for t in analysis["top_topics"][:3]],
            "key_entities": [e["name"] for e in analysis["top_entities"][:5]],
            "n_sources": analysis["n_analyzed"],
            "timestamp": datetime.utcnow().isoformat(),
        }


# Module-level instance
nlp_analyzer = CryptoNLPAnalyzer()
