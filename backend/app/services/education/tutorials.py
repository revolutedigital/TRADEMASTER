"""Education center: interactive tutorials, quizzes, and certification."""

from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Tutorial:
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    category: str
    content_markdown: str
    estimated_minutes: int
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class QuizQuestion:
    id: str
    question: str
    options: list[str]
    correct_index: int
    explanation: str


@dataclass
class Quiz:
    id: str
    title: str
    tutorial_id: str
    questions: list[QuizQuestion]
    passing_score: float = 0.7


class EducationCenter:
    """Interactive learning platform for traders."""

    TUTORIALS = [
        Tutorial(
            id="intro-01", title="Introduction to Cryptocurrency Trading",
            description="Learn the basics of buying and selling digital assets",
            difficulty=DifficultyLevel.BEGINNER, category="fundamentals",
            content_markdown="# Introduction to Crypto Trading\n\nCryptocurrency trading involves...",
            estimated_minutes=15,
        ),
        Tutorial(
            id="risk-01", title="Understanding Risk Management",
            description="Learn how to protect your capital with proper risk management",
            difficulty=DifficultyLevel.BEGINNER, category="risk_management",
            content_markdown="# Risk Management\n\nRisk management is the most important...",
            estimated_minutes=20,
            prerequisites=["intro-01"],
        ),
        Tutorial(
            id="ta-01", title="Technical Analysis Fundamentals",
            description="Learn to read charts and identify patterns",
            difficulty=DifficultyLevel.INTERMEDIATE, category="technical_analysis",
            content_markdown="# Technical Analysis\n\nTechnical analysis uses price and volume...",
            estimated_minutes=30,
            prerequisites=["intro-01"],
        ),
        Tutorial(
            id="ml-01", title="How AI/ML Models Generate Signals",
            description="Understand how TradeMaster uses machine learning",
            difficulty=DifficultyLevel.ADVANCED, category="ml_trading",
            content_markdown="# ML Trading Signals\n\nTradeMaster uses LSTM and XGBoost...",
            estimated_minutes=25,
            prerequisites=["ta-01"],
        ),
        Tutorial(
            id="backtest-01", title="Backtesting Strategies",
            description="Learn to validate strategies with historical data",
            difficulty=DifficultyLevel.INTERMEDIATE, category="backtesting",
            content_markdown="# Backtesting\n\nBacktesting is the process of testing...",
            estimated_minutes=20,
            prerequisites=["ta-01", "risk-01"],
        ),
    ]

    QUIZZES = [
        Quiz(
            id="quiz-risk-01", title="Risk Management Quiz", tutorial_id="risk-01",
            questions=[
                QuizQuestion("q1", "What is the recommended maximum risk per trade?", ["1-2%", "5-10%", "20-30%", "50%"], 0, "Most professionals risk 1-2% per trade."),
                QuizQuestion("q2", "What does a circuit breaker do?", ["Increases leverage", "Stops trading when losses exceed threshold", "Adds more capital", "Changes strategy"], 1, "Circuit breakers halt trading to prevent catastrophic losses."),
                QuizQuestion("q3", "What is Value at Risk (VaR)?", ["Expected profit", "Maximum possible loss at a confidence level", "Average return", "Fee estimate"], 1, "VaR estimates the maximum loss at a given confidence level."),
            ],
            passing_score=0.67,
        ),
    ]

    def get_tutorials(self, category: str | None = None, difficulty: DifficultyLevel | None = None) -> list[Tutorial]:
        tutorials = self.TUTORIALS
        if category:
            tutorials = [t for t in tutorials if t.category == category]
        if difficulty:
            tutorials = [t for t in tutorials if t.difficulty == difficulty]
        return tutorials

    def get_tutorial(self, tutorial_id: str) -> Tutorial | None:
        return next((t for t in self.TUTORIALS if t.id == tutorial_id), None)

    def get_quiz(self, quiz_id: str) -> Quiz | None:
        return next((q for q in self.QUIZZES if q.id == quiz_id), None)

    def grade_quiz(self, quiz_id: str, answers: dict[str, int]) -> dict:
        quiz = self.get_quiz(quiz_id)
        if not quiz:
            return {"error": "Quiz not found"}

        correct = 0
        total = len(quiz.questions)
        results = []

        for question in quiz.questions:
            user_answer = answers.get(question.id)
            is_correct = user_answer == question.correct_index
            if is_correct:
                correct += 1
            results.append({
                "question_id": question.id,
                "correct": is_correct,
                "correct_answer": question.correct_index,
                "explanation": question.explanation,
            })

        score = correct / total if total > 0 else 0
        passed = score >= quiz.passing_score

        return {
            "quiz_id": quiz_id,
            "score": round(score, 2),
            "passed": passed,
            "correct": correct,
            "total": total,
            "results": results,
            "certificate": passed,
        }


education_center = EducationCenter()
